def interaction(recsys, environment, n=100):
    """ Interaction between RecSystem and Environment. """
    for _ in range(n):
        user, actions = environment.current_state()
        action = recsys.best_action(user, actions)
        payoff = environment.reaction(user['id'], action['id'])
        recsys.update(user, action, payoff)

    return None


# class Experiment:
#     def interaction(self, n=100):
#         interaction(self.recsys, self.environment, n=n)
#         return None
