import unittest
import numpy as np

from interaction import interaction
from environment import ConstEnvironment
from recsystem import LinUCB


class TestLinUCB(unittest.TestCase):
    def setUp(self):

        def formatter(arr): return [{'id': i, 'features': np.array(x)}
                                    for i, x in enumerate(arr)]

        self.users = formatter(
            [[0.58949733, 0.75165592, 0.13686766],
             [0.92270089, 0.98501207, 0.12552882],
             [0.74586265, 0.09031383, 0.0046303],
             [0.43470956, 0.15848989, 0.55960944]]
        )

        self.docs = formatter(
            [[0.10455431, 0.62316076],
             [0.0512237, 0.74471496],
             [0.70174098, 0.63902647],
             [0.50405523, 0.83093542]]
        )
        reactions = {i: 1/(i + 1) for i in range(4)}

        self.env = ConstEnvironment(self.users, self.docs, reactions)

        self.rec = LinUCB(3, 2)

    def test_t001_init(self):
        self.assertEqual(np.sum(self.rec.A0), 3)
        self.assertEqual(np.sum(self.rec.b0), 0)

    def test_t002_work_with_new_action(self):
        A, B, b = self.rec.get_action_param(self.docs[-1]['id'])
        self.assertEqual(np.sum(A), 2)
        self.assertEqual(np.sum(B), 0)
        self.assertEqual(np.sum(b), 0)

    def test_t003_interaction(self):
        interaction(self.rec, self.env, n=4)
        self.assertTrue((self.rec.A0 != np.eye(3)).any())
        self.assertTrue((self.rec.b0 != np.zeros(3)).any())

    def test_t004_hist_learn(self):
        hist = [(self.users[0], self.docs[0], -1),
                (self.users[1], self.docs[1], 1),
                (self.users[2], self.docs[2], -1)
                ]
        self.rec.learn(hist)
        b = self.rec.best_action(self.users[3], self.docs)
        self.assertEqual(b['id'], 1)


if __name__ == "__main__":
    unittest.main()
