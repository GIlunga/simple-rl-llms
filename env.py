import random
import re
from dataclasses import dataclass

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


@dataclass
class StepResult:
    observation: str
    terminated: bool
    truncated: bool
    guess: int | None


@dataclass
class RolloutMetrics:
    reward: float
    won: bool
    turn_count: int
    out_of_range_count: int
    repeated_count: int
    direction_events: int
    wrong_direction_count: int
    reward_by_max_turns: dict[int, float]


class GuessTheNumberEnv:
    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        self.target: int | None = None
        self.turn_count: int = 0
        self.previous_guesses: set[int] = set()
        self.guesses: list[int | None] = []

    @property
    def min_number(self) -> int:
        return self.config.min_number

    @property
    def max_number(self) -> int:
        return self.config.max_number

    @property
    def system_prompt(self) -> str:
        return (
            "You are playing Guess The Number with the user. The user has a target number between "
            f"{self.min_number} and {self.max_number} (inclusive) and you have to guess it as fast as possible."
            "When you enter a guess, the user will tell you if the target number is 'higher' or 'lower'. "
            "When answering, only the number that is wrapped inside \\boxed{} will be considered as your guess, "
            "for example, \\boxed{1}. Follow that exact format for your final answer."
        )

    @property
    def initial_message(self) -> str:
        return "Enter your first guess to start the game!"

    def reset(self) -> str:
        self.target = random.randint(self.min_number, self.max_number)
        self.turn_count = 0
        self.previous_guesses = set()
        self.guesses = []
        return self.initial_message

    def clone(self) -> "GuessTheNumberEnv":
        """Copy the current state (target included) so rollouts share a target."""
        new = GuessTheNumberEnv(self.config)
        new.target = self.target
        new.turn_count = self.turn_count
        new.previous_guesses = set(self.previous_guesses)
        new.guesses = list(self.guesses)
        return new

    def step(self, action: str) -> StepResult:
        if self.target is None:
            raise RuntimeError("GuessTheNumberEnv.step() called before reset()")

        self.turn_count += 1
        matches = BOXED_GUESS_PATTERN.findall(action)
        guess = int(matches[-1]) if matches else None

        if guess is None:
            self.guesses.append(None)
            obs = f"At turn {self.turn_count}, you did not provide a valid guess."
            return StepResult(obs, terminated=True, truncated=self.turn_count == self.config.max_turns, guess=None)

        if guess < self.min_number or guess > self.max_number:
            self.guesses.append(guess)
            obs = f"At turn {self.turn_count}, you guessed {guess}, which is outside the range specified."
        elif guess in self.previous_guesses:
            self.guesses.append(guess)
            obs = f"At turn {self.turn_count}, you guessed {guess}, which has been already guessed before."
        else:
            self.previous_guesses.add(guess)
            self.guesses.append(guess)
            if guess == self.target:
                obs = f"Congratulations! You guessed the correct number {self.target} in {self.turn_count} turns."
                return StepResult(obs, terminated=True, truncated=False, guess=guess)
            hint = "lower" if guess > self.target else "higher"
            obs = f"At turn {self.turn_count}, you guessed {guess}, and the target number is {hint} than {guess}."

        if self.turn_count >= self.config.max_turns:
            obs = "You have reached the maximum number of turns."
            return StepResult(obs, terminated=True, truncated=True, guess=guess)
        return StepResult(obs, terminated=False, truncated=False, guess=guess)

    def score(self, budgets: tuple[int, ...] | list[int]) -> dict[int, float]:
        """Reward for every turn budget, derived by truncating the recorded guesses."""
        sorted_budgets = sorted(budgets)
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
                    terminal_reward = self.config.format_error_reward
                elif guess < self.min_number or guess > self.max_number:
                    last_guess, last_guess_invalid = guess, True
                elif guess in seen:
                    last_guess, last_guess_invalid = guess, True
                elif guess == self.target:
                    terminal_reward = 1.0
                else:
                    last_guess, last_guess_invalid = guess, False
                    seen.add(guess)

            if turn in budget_set:
                results[turn] = (
                    terminal_reward
                    if terminal_reward is not None
                    else self._truncation_reward(last_guess, last_guess_invalid)
                )

        return results

    def _truncation_reward(self, last_guess: int | None, last_guess_invalid: bool) -> float:
        if last_guess is None:
            return self.config.format_error_reward

        if last_guess_invalid:
            return self.config.invalid_action_reward

        if self.config.reward_type == "binary":
            return 0.0

        # Dense case, give partial credit on the final guess
        return 1.0 - abs(last_guess - self.target) / (self.max_number - self.min_number)

    def metrics(self, reward_budget: int) -> RolloutMetrics:
        """Per-rollout count metrics and the training reward at ``reward_budget``."""
        guesses = [g for g in self.guesses if g is not None]
        reward_by_max_turns = self.score(self.config.reward_breakdown_turns)
        reward = reward_by_max_turns[reward_budget]

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
            reward=reward,
            won=reward == 1,
            turn_count=self.turn_count,
            out_of_range_count=out_of_range_count,
            repeated_count=repeated_count,
            direction_events=direction_events,
            wrong_direction_count=wrong_direction_count,
            reward_by_max_turns=reward_by_max_turns,
        )


def compute_env_metrics(
    metrics: list[RolloutMetrics],
    num_completions_per_prompt: int,
) -> dict[str, float | int]:
    """Aggregate per-rollout env metrics into the wandb ``dataset_metrics`` dict."""
    total_size = len(metrics)

    out: dict[str, float | int] = {
        "Quality/out of range turn count": sum(m.out_of_range_count for m in metrics),
        "Quality/repeated turn count": sum(m.repeated_count for m in metrics),
        "Quality/wrong direction rate": sum(m.wrong_direction_count for m in metrics)
        / max(sum(m.direction_events for m in metrics), 1),
    }

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
            f"Rewards@{budget}/avg": mean,
            f"Rewards@{budget}/std": variance**0.5,
            f"Rewards@{budget}/zero rate": sum(1 for r in budget_rewards if r <= 0) / total_size,
            f"Rewards@{budget}/one rate": sum(1 for r in budget_rewards if r == 1) / total_size,
            f"Rewards@{budget}/zero group rate": sum(1 for g in groups if all(r <= 0 for r in g)) / num_groups,
            f"Rewards@{budget}/one group rate": sum(1 for g in groups if all(r == 1 for r in g)) / num_groups,
            f"Rewards@{budget}/passing group rate": sum(1 for g in groups if any(r == 1 for r in g)) / num_groups,
        }

    return out
