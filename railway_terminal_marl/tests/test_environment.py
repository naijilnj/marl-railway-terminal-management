import unittest

from env import Platform, RailwayTerminusEnv, ScenarioConfig, TrainSpec, traffic_scenario
from env.allocation import maximum_weight_matching
from training.evaluation import evaluate_episode


def scenario(platforms, trains, horizon=12, max_queue=8, name="test"):
    return ScenarioConfig(
        horizon=horizon,
        max_queue=max_queue,
        platforms=platforms,
        trains=trains,
        name=name,
    )


def train(train_id, arrival=0, category="Express", dwell=1, priority=0):
    return TrainSpec(
        id=train_id,
        scheduled_arrival=arrival,
        dwell_time=dwell,
        length_class="short",
        electrified_required=False,
        priority=priority,
        category=category,
    )


class EnvironmentTests(unittest.TestCase):
    def test_turnaround_buffer_blocks_reuse_until_ready_time(self):
        platform = Platform(
            id=0, length_class="long", electrified=True, turnaround_buffer=2
        )
        env = RailwayTerminusEnv(
            scenario([platform], [train(0), train(1, arrival=1)], horizon=8)
        )
        env.reset(seed=11)

        env.step({"platform_0": 0})
        self.assertEqual(env.platform_ready_at["platform_0"], 3)
        env.step({"platform_0": 0})
        env.step({"platform_0": 0})
        self.assertEqual(env.trains_served, 1)
        env.step({"platform_0": 0})
        self.assertEqual(env.trains_served, 2)

    def test_queue_overflow_records_diversion_instead_of_dropping_train(self):
        platform = Platform(id=0, length_class="long", electrified=True)
        env = RailwayTerminusEnv(
            scenario([platform], [train(0), train(1), train(2)], max_queue=1)
        )
        env.reset(seed=3)

        self.assertEqual(len(env.queue), 1)
        self.assertEqual(env.train_states, {0: "waiting", 1: "diverted", 2: "diverted"})
        self.assertEqual(sum(event["type"] == "diverted" for event in env.events), 2)

    def test_closed_platform_cannot_allocate(self):
        platform = Platform(
            id=0,
            length_class="long",
            electrified=True,
            closure_intervals=[(0, 2)],
        )
        env = RailwayTerminusEnv(scenario([platform], [train(0)], horizon=6))
        env.reset()

        env.step({"platform_0": 0})
        self.assertEqual(env.trains_served, 0)
        env.step({"platform_0": 0})
        self.assertEqual(env.trains_served, 0)
        env.step({"platform_0": 0})
        self.assertEqual(env.trains_served, 1)

    def test_matching_respects_categories_and_approach_capacity(self):
        platforms = [
            Platform(
                id=0, length_class="long", electrified=True,
                compatible_categories=["Express"], approach_group="shared",
            ),
            Platform(
                id=1, length_class="long", electrified=True,
                compatible_categories=["Express"], approach_group="shared",
            ),
        ]
        env = RailwayTerminusEnv(
            scenario(platforms, [train(0), train(1)], horizon=5),
            coordination="matching",
        )
        env.reset()

        env.step({"platform_0": 2, "platform_1": 2})
        allocations = [event for event in env.events if event["type"] == "allocate"]
        self.assertEqual(len(allocations), 1)
        self.assertEqual(len({event["train"] for event in allocations}), 1)

    def test_negative_action_is_counted_and_does_not_claim_last_train(self):
        platform = Platform(id=0, length_class="long", electrified=True)
        env = RailwayTerminusEnv(scenario([platform], [train(0)]))
        env.reset()

        env.step({"platform_0": -1})
        self.assertEqual(env.invalid_empty_actions, 1)
        self.assertEqual(env.trains_served, 0)
        self.assertEqual(env.train_states[0], "waiting")

    def test_lowest_id_and_seeded_random_conflict_resolution(self):
        platforms = [
            Platform(id=0, length_class="long", electrified=True, approach_group="a"),
            Platform(id=1, length_class="long", electrified=True, approach_group="b"),
        ]
        baseline = RailwayTerminusEnv(
            scenario(platforms, [train(0)]), coordination="lowest_id"
        )
        baseline.reset(seed=7)
        baseline.step({"platform_0": 0, "platform_1": 0})
        self.assertEqual(
            next(event["platform"] for event in baseline.events if event["type"] == "allocate"),
            0,
        )

        winners = []
        for _ in range(2):
            randomized = RailwayTerminusEnv(
                scenario(platforms, [train(0)]), coordination="random"
            )
            randomized.reset(seed=23)
            randomized.step({"platform_0": 0, "platform_1": 0})
            winners.append(
                next(event["platform"] for event in randomized.events if event["type"] == "allocate")
            )
        self.assertEqual(winners[0], winners[1])

    def test_matching_finds_global_maximum_not_greedy_first_edge(self):
        weights = [
            [10.0, 9.0],
            [9.0, 0.0],
        ]
        self.assertEqual(set(maximum_weight_matching(weights)), {(0, 1), (1, 0)})

    def test_traffic_profiles_are_reproducible_and_contain_disruptions(self):
        first = traffic_scenario("platform_disruption", seed=41)
        second = traffic_scenario("platform_disruption", seed=41)
        self.assertEqual(first.trains, second.trains)
        self.assertEqual(
            [platform.closure_intervals for platform in first.platforms],
            [platform.closure_intervals for platform in second.platforms],
        )
        self.assertTrue(any(platform.closure_intervals for platform in first.platforms))
        self.assertTrue(any(train.unexpected for train in first.trains))
        self.assertGreaterEqual(first.horizon, first.arrival_horizon)

    def test_low_traffic_baseline_can_dispatch_last_scheduled_arrival(self):
        test_scenario = traffic_scenario("low_traffic", seed=10001)
        metrics = evaluate_episode(
            test_scenario, "fcfs", evaluation_seed=10001
        )
        self.assertEqual(metrics["trains_served"], len(test_scenario.trains))
        self.assertEqual(metrics["stranded_trains"], 0)


if __name__ == "__main__":
    unittest.main()
