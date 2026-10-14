import random
import re
from dataclasses import dataclass
from enum import StrEnum

BOXED_GUESS_PATTERN = re.compile(r"\\boxed\{(-?\d+)\}")


@dataclass(frozen=True)
class EnvConfig:
    min_number: int
    max_number: int
    max_turns: int
    reward_type: str
    format_error_reward: float
    invalid_action_reward: float
    reward_breakdown_turns: tuple[int, ...]


class TurnOutcome(StrEnum):
    FORMAT_ERROR = "format_error"
    OUT_OF_RANGE = "out_of_range"
    REPEATED = "repeated"
    CORRECT = "correct"
    DIRECTIONAL = "directional"


@dataclass
class TurnRecord:
    guess: int | None
    outcome: TurnOutcome


@dataclass
class StepResult:
    observation: str
    terminated: bool
    truncated: bool


@dataclass
class RolloutMetrics:
    won: bool
    turn_count: int
    out_of_range_count: int
    repeated_count: int
    direction_events: int
    wrong_direction_count: int
    reward_by_max_turns: dict[int, float]


class GuessTheNumberEnv:
    def __init__(self, config: EnvConfig, target: int) -> None:
        self.config = config
        self.target = target

        self.initial_message = "Enter your first guess to start the game!"
        self.system_prompt = "You are playing Guess The Number with the user. The user has a target number between " \
        f"{config.min_number} and {config.max_number} (inclusive) and you have to guess it as fast as possible."
        "When you enter a guess, the user will tell you if the target number is 'higher' or 'lower'. "
        "When answering, only the number that is wrapped inside \\boxed{} will be considered as your guess, "
        "for example, \\boxed{1}. Follow that exact format for your final answer."

        self.turn_count: int = 0
        self.previous_guesses: set[int] = set()
        self.turns: list[TurnRecord] = []
        self.terminal_turn: int | None = None

        # Quality metric counters, maintained incrementally as the game is played.
        self.out_of_range_count: int = 0
        self.repeated_count: int = 0
        self.direction_events: int = 0
        self.wrong_direction_count: int = 0
        self.prev_was_directional: bool = False
        self.prev_directional_guess: int = 0

    @classmethod
    def spawn(cls, config: EnvConfig, num_envs: int, target: int | None = None) -> list["GuessTheNumberEnv"]:
        """Return ``num_envs`` fresh envs sharing the target number (random if not given)."""
        if target is None:
            target = random.randint(config.min_number, config.max_number)
        return [cls(config, target) for _ in range(num_envs)]

    def _update_direction_metrics(self, guess: int | None) -> None:
        if guess is None or not self.prev_was_directional:
            return
        self.direction_events += 1
        hint_said_higher = self.prev_directional_guess < self.target
        if (hint_said_higher and guess < self.prev_directional_guess) or (
            not hint_said_higher and guess > self.prev_directional_guess
        ):
            self.wrong_direction_count += 1

    def _record(self, outcome: TurnOutcome, guess: int | None) -> None:
        self.turns.append(TurnRecord(guess=guess, outcome=outcome))
        if guess is not None and self.config.min_number <= guess <= self.config.max_number:
            self.previous_guesses.add(guess)

        if outcome is TurnOutcome.OUT_OF_RANGE:
            self.out_of_range_count += 1
        elif outcome is TurnOutcome.REPEATED:
            self.repeated_count += 1

        if outcome is TurnOutcome.DIRECTIONAL:
            self.prev_was_directional = True
            self.prev_directional_guess = guess
        else:
            self.prev_was_directional = False

    def step(self, action: str) -> StepResult:
        self.turn_count += 1
        matches = BOXED_GUESS_PATTERN.findall(action)
        guess = int(matches[-1]) if matches else None  # Always take last match

        self._update_direction_metrics(guess)

        # Immediately end if no guess is given
        if guess is None:
            obs = f"At turn {self.turn_count}, you did not provide a valid guess."
            self._record(TurnOutcome.FORMAT_ERROR, guess)
            self.terminal_turn = self.turn_count
            return StepResult(obs, terminated=True, truncated=self.turn_count == self.config.max_turns)

        if guess < self.config.min_number or guess > self.config.max_number:
            outcome = TurnOutcome.OUT_OF_RANGE
            obs = f"At turn {self.turn_count}, you guessed {guess}, which is outside the range specified."
        elif guess in self.previous_guesses:
            outcome = TurnOutcome.REPEATED
            obs = f"At turn {self.turn_count}, you guessed {guess}, which has been already guessed before."
        elif guess == self.target:
            outcome = TurnOutcome.CORRECT
            obs = f"Congratulations! You guessed the correct number {self.target} in {self.turn_count} turns."
        else:
            outcome = TurnOutcome.DIRECTIONAL
            hint = "lower" if guess > self.target else "higher"
            obs = f"At turn {self.turn_count}, you guessed {guess}, and the target number is {hint} than {guess}."

        self._record(outcome, guess)

        if outcome is TurnOutcome.CORRECT:
            self.terminal_turn = self.turn_count
            return StepResult(obs, terminated=True, truncated=False)

        if self.turn_count >= self.config.max_turns:
            return StepResult("You have reached the maximum number of turns.", terminated=True, truncated=True)
        return StepResult(obs, terminated=False, truncated=False)

    def _turn_reward(self, record: TurnRecord) -> float:
        """Reward of a turn if the episode ended on it."""
        if record.outcome is TurnOutcome.FORMAT_ERROR:
            return self.config.format_error_reward
        if record.outcome is TurnOutcome.CORRECT:
            return 1.0
        if record.outcome in (TurnOutcome.OUT_OF_RANGE, TurnOutcome.REPEATED):
            return self.config.invalid_action_reward
        if self.config.reward_type == "binary":
            return 0.0

        # Dense distance reward
        return 1.0 - abs(record.guess - self.target) / (self.config.max_number - self.config.min_number)

    def _rewards_by_max_turns(self) -> dict[int, float]:
        """Reward for every turn budget: the reward of the turn at ``min(budget, terminal_turn)``."""
        if not self.turns:
            return {budget: self.config.format_error_reward for budget in self.config.reward_breakdown_turns}

        end = self.terminal_turn if self.terminal_turn is not None else len(self.turns)
        return {
            budget: self._turn_reward(self.turns[min(budget, end) - 1])
            for budget in self.config.reward_breakdown_turns
        }

    def rollout_metrics(self) -> RolloutMetrics:
        """Per-rollout count metrics and rewards for every turn budget."""
        reward_by_max_turns = self._rewards_by_max_turns()

        return RolloutMetrics(
            won=reward_by_max_turns[max(self.config.reward_breakdown_turns)] == 1.0,
            turn_count=self.turn_count,
            out_of_range_count=self.out_of_range_count,
            repeated_count=self.repeated_count,
            direction_events=self.direction_events,
            wrong_direction_count=self.wrong_direction_count,
            reward_by_max_turns=reward_by_max_turns,
        )


def aggregate_quality_metrics(metrics: list[RolloutMetrics], prefix: str = "") -> dict[str, float | int]:
    return {
        f"{prefix}Quality/out of range turn count": sum(m.out_of_range_count for m in metrics),
        f"{prefix}Quality/repeated turn count": sum(m.repeated_count for m in metrics),
        f"{prefix}Quality/wrong direction rate": sum(m.wrong_direction_count for m in metrics)
        / max(sum(m.direction_events for m in metrics), 1),
    }


def aggregate_env_metrics(
    metrics: list[RolloutMetrics],
    num_completions_per_prompt: int,
    prefix: str = "",
) -> dict[str, float | int]:
    """Aggregate a batch of rollouts into reward and quality statistics.

    ``metrics`` must be ordered so that each consecutive block of ``num_completions_per_prompt``
    rollouts belongs to the same prompt (as produced by ``get_rollouts``).
    """
    total_size = len(metrics)

    out: dict[str, float | int] = aggregate_quality_metrics(metrics, prefix)

    # Reward metrics for every turn budget
    budgets = sorted({budget for m in metrics for budget in m.reward_by_max_turns})
    for budget in budgets:
        budget_rewards = [m.reward_by_max_turns[budget] for m in metrics]
        mean = sum(budget_rewards) / total_size
        variance = sum((r - mean) ** 2 for r in budget_rewards) / (total_size - 1) if total_size > 1 else 0.0

        num_groups = total_size // num_completions_per_prompt
        groups = [
            budget_rewards[i * num_completions_per_prompt : (i + 1) * num_completions_per_prompt]
            for i in range(num_groups)
        ]

        out |= {
            f"{prefix}Rewards@{budget}/avg": mean,
            f"{prefix}Rewards@{budget}/std": variance**0.5,
            f"{prefix}Rewards@{budget}/zero rate": sum(1 for r in budget_rewards if r <= 0) / total_size,
            f"{prefix}Rewards@{budget}/one rate": sum(1 for r in budget_rewards if r == 1) / total_size,
            f"{prefix}Rewards@{budget}/zero group rate": sum(1 for g in groups if all(r <= 0 for r in g)) / num_groups,
            f"{prefix}Rewards@{budget}/one group rate": sum(1 for g in groups if all(r == 1 for r in g)) / num_groups,
            f"{prefix}Rewards@{budget}/passing group rate": sum(1 for g in groups if any(r == 1 for r in g))
            / num_groups,
        }

    return out
