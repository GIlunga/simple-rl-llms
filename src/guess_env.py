import copy
import random
import re
from dataclasses import dataclass

BOXED_GUESS_PATTERN = re.compile(r"\\boxed\{(-?\d+)\}")


@dataclass
class StepResult:
    observation: str
    terminated: bool
    truncated: bool
    guess: int | None


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
    initial_message = "Enter your first guess to start the game!"

    def __init__(
        self,
        min_number: int,
        max_number: int,
        max_turns: int,
        reward_type: str,
        format_error_reward: float,
        invalid_action_reward: float,
        reward_breakdown_turns: tuple[int, ...],
    ) -> None:
        self.min_number = min_number
        self.max_number = max_number
        self.max_turns = max_turns
        self.reward_type = reward_type
        self.format_error_reward = format_error_reward
        self.invalid_action_reward = invalid_action_reward
        self.reward_breakdown_turns = reward_breakdown_turns

        self.target: int | None = None
        self.turn_count: int = 0
        self.previous_guesses: set[int] = set()
        self.guesses: list[int | None] = []

    @property
    def system_prompt(self) -> str:
        return (
            "You are playing Guess The Number with the user. The user has a target number between "
            f"{self.min_number} and {self.max_number} (inclusive) and you have to guess it as fast as possible."
            "When you enter a guess, the user will tell you if the target number is 'higher' or 'lower'. "
            "When answering, only the number that is wrapped inside \\boxed{} will be considered as your guess, "
            "for example, \\boxed{1}. Follow that exact format for your final answer."
        )

    def reset(self, target: int | None = None) -> str:
        self.target = target if target is not None else random.randint(self.min_number, self.max_number)
        self.turn_count = 0
        self.previous_guesses = set()
        self.guesses = []
        return self.initial_message

    def clone(self) -> "GuessTheNumberEnv":
        new = copy.copy(self)
        new.previous_guesses = set(self.previous_guesses)
        new.guesses = list(self.guesses)
        return new

    def _record(self, guess: int | None) -> None:
        """Append a guess to the history and remember valid guesses for repeat detection."""
        self.guesses.append(guess)
        if guess is not None and self.min_number <= guess <= self.max_number:
            self.previous_guesses.add(guess)

    def step(self, action: str) -> StepResult:
        if self.target is None:
            raise RuntimeError("GuessTheNumberEnv.step() called before reset()")

        self.turn_count += 1
        matches = BOXED_GUESS_PATTERN.findall(action)
        guess = int(matches[-1]) if matches else None

        if guess is None:
            self._record(None)
            obs = f"At turn {self.turn_count}, you did not provide a valid guess."
            return StepResult(obs, terminated=True, truncated=self.turn_count == self.max_turns, guess=None)

        if guess < self.min_number or guess > self.max_number:
            self._record(guess)
            obs = f"At turn {self.turn_count}, you guessed {guess}, which is outside the range specified."
        elif guess in self.previous_guesses:
            self._record(guess)
            obs = f"At turn {self.turn_count}, you guessed {guess}, which has been already guessed before."
        elif guess == self.target:
            self._record(guess)
            obs = f"Congratulations! You guessed the correct number {self.target} in {self.turn_count} turns."
            return StepResult(obs, terminated=True, truncated=False, guess=guess)
        else:
            self._record(guess)
            hint = "lower" if guess > self.target else "higher"
            obs = f"At turn {self.turn_count}, you guessed {guess}, and the target number is {hint} than {guess}."

        if self.turn_count >= self.max_turns:
            return StepResult(
                "You have reached the maximum number of turns.",
                terminated=True,
                truncated=True,
                guess=guess,
            )
        return StepResult(obs, terminated=False, truncated=False, guess=guess)

    def _rewards_by_max_turns(self) -> dict[int, float]:
        """Reward for every turn budget, derived by truncating the recorded guesses."""
        sorted_budgets = sorted(self.reward_breakdown_turns)
        budget_set = set(sorted_budgets)

        results: dict[int, float] = {}
        terminal_reward: float | None = None
        last_guess: int | None = None  # last parsed guess (valid or invalid)
        last_guess_invalid = False
        seen: set[int] = set()

        for turn in range(1, sorted_budgets[-1] + 1):
            if terminal_reward is None and turn <= len(self.guesses):
                guess = self.guesses[turn - 1]
                if guess is None:
                    terminal_reward = self.format_error_reward
                elif guess < self.min_number or guess > self.max_number:
                    last_guess, last_guess_invalid = guess, True
                elif guess in seen:
                    last_guess, last_guess_invalid = guess, True
                elif guess == self.target:
                    terminal_reward = 1.0
                else:
                    last_guess, last_guess_invalid = guess, False
                    seen.add(guess)

            # A terminal reward ends the trajectory: every remaining budget inherits it.
            if terminal_reward is not None:
                for budget in sorted_budgets:
                    results.setdefault(budget, terminal_reward)
                return results

            if turn in budget_set:
                results[turn] = self._truncation_reward(last_guess, last_guess_invalid)

        return results

    def _truncation_reward(self, last_guess: int | None, last_guess_invalid: bool) -> float:
        if last_guess is None:
            return self.format_error_reward

        if last_guess_invalid:
            return self.invalid_action_reward

        if self.reward_type == "binary":
            return 0.0

        # Dense case, give partial credit on the final guess
        return 1.0 - abs(last_guess - self.target) / (self.max_number - self.min_number)

    def rollout_metrics(self) -> RolloutMetrics:
        """Per-rollout count metrics and rewards for every turn budget."""
        reward_by_max_turns = self._rewards_by_max_turns()
        max_budget = max(self.reward_breakdown_turns)

        guesses = [g for g in self.guesses if g is not None]

        out_of_range_count = 0
        direction_events = 0
        wrong_direction_count = 0

        seen: set[int] = set()
        prev_was_directional = False
        prev_guess = 0

        for g in guesses:
            if g < self.min_number or g > self.max_number:
                out_of_range_count += 1

            if prev_was_directional:
                direction_events += 1
                hint_said_higher = prev_guess < self.target
                if (hint_said_higher and g < prev_guess) or (not hint_said_higher and g > prev_guess):
                    wrong_direction_count += 1

            if g < self.min_number or g > self.max_number:
                prev_was_directional = False
            elif g in seen:
                prev_was_directional = False
            else:
                prev_was_directional = True
                prev_guess = g
                seen.add(g)

        repeated_count = len(guesses) - len(set(guesses))

        return RolloutMetrics(
            won=reward_by_max_turns[max_budget] == 1.0,
            turn_count=self.turn_count,
            out_of_range_count=out_of_range_count,
            repeated_count=repeated_count,
            direction_events=direction_events,
            wrong_direction_count=wrong_direction_count,
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
            f"{prefix}Rewards@{budget}/zero group rate": sum(1 for g in groups if all(r <= 0 for r in g))
            / num_groups,
            f"{prefix}Rewards@{budget}/one group rate": sum(1 for g in groups if all(r == 1 for r in g)) / num_groups,
            f"{prefix}Rewards@{budget}/passing group rate": sum(1 for g in groups if any(r == 1 for r in g))
            / num_groups,
        }

    return out
