# user = {'id': int, 'features': vect}
# action = {'id': int, 'features': vect}

import numpy as np


class RecSystem:
    """ An interface for online recommend systems. """

    def predict(self, user, documents):
        # Выдача должна быть в виде списка словарей с полями
        # [ {id: int, range: double} ]
        raise NotImplementedError

    def best_action(self, user, documents):
        arr = self.predict(user, documents)
        best = sorted(arr, key=lambda x: x['range'], reverse=True)[0]
        return best

    def update(self, user, action, payoff):
        raise NotImplementedError

    def learn(self, hist):
        for user, action, payoff in hist:
            self.update(user, action, payoff)


class LinUCB(RecSystem):
    def __init__(self, k, d, alpha=1):
        self.k = k  # dimention of user-acticle combination features
        self.d = d  # dimention of acticle featrues
        self.alpha = alpha

        self.A0 = np.eye(k)
        self.b0 = np.zeros(k)

        self.A = dict()
        self.B = dict()
        self.b = dict()

    def get_action_param(self, action_id):
        """ Возвращаем триплет параметров A_action, B_action, b_action. """

        if self.A.get(action_id) is None:
            self.A[action_id] = np.eye(self.d)
            self.B[action_id] = np.zeros((self.d, self.k))
            self.b[action_id] = np.zeros(self.d)

        return self.A[action_id], self.B[action_id], self.b[action_id]

    def update(self, user, action, payoff):
        """ Обновить параметры исходя из опыта: юзеру user показали action и
        получили отклик payoff. """

        z, x = user['features'], action['features']
        A, B, b = self.get_action_param(action['id'])
        BA = B.T @ np.linalg.inv(A)

        self.A0 += BA @ B
        self.b0 += BA @ b

        def f(vect): return np.array([vect]).T @ np.array([vect])
        self.A[action['id']] += np.array([x]).T @ np.array([x])
        self.B[action['id']] += np.array([x]).T @ np.array([z])
        self.b[action['id']] += payoff * x
        A, B, b = self.get_action_param(action['id'])
        BA = B.T @ np.linalg.inv(A)

        self.A0 += np.array([z]).T @ np.array([z]) - BA @ B
        self.b0 += payoff * z - BA @ b

        return None

    def best_action(self, user, documents):

        A0i = np.linalg.inv(self.A0)
        beta = A0i @ self.b0
        best_action, best_p = None, - np.inf
        for action in documents:
            A, B, b = self.get_action_param(action['id'])
            Ai = np.linalg.inv(A)
            teta = Ai @ (b - B @ beta)
            z, x = user['features'], action['features']
            s = z @ A0i @ z + x @ Ai @ x
            s += (x.T @ Ai @ B - 2 * z.T) @ A0i @ B.T @ Ai @ x
            p = z.T @ beta + x.T @ teta + self.alpha * np.sqrt(s)

            if p > best_p:
                best_action = action
                best_p = p

        return best_action
