import unittest
import numpy as np

from interaction import interaction
from environment import ConstEnvironment
from recsystem import LinUCB


class TestLinUCB(unittest.TestCase):
    def setUp(self):

        users = [{'id': i, 'features': np.array(
            [1, 0])} for i in range(3)]
        self.docs = [{'id': i, 'features': np.array(
            [1, 1, 0, 0])} for i in range(9)]
        reactions = {i: 1/(i + 1) for i in range(3)}

        self.env = ConstEnvironment(users, self.docs, reactions)

        self.rec = LinUCB(2, 4)

    def test_t001_init(self):
        self.assertEqual(np.sum(self.rec.A0), 2)
        self.assertEqual(np.sum(self.rec.b0), 0)

    def test_t002_work_with_new_action(self):
        A, B, b = self.rec.get_action_param(self.docs[8]['id'])
        self.assertEqual(np.sum(A), 4)
        self.assertEqual(np.sum(B), 0)
        self.assertEqual(np.sum(b), 0)

    @unittest.skip('asdf')
    def test_t003_interaction(self):
        interaction(self.rec, self.env, n=4)

    def test_t004_after_interaction(self):
        interaction(self.rec, self.env, n=4)
        self.assertTrue((self.rec.A0 != np.eye(2)).any())
        self.assertTrue((self.rec.b0 != np.zeros(2)).any())


if __name__ == "__main__":
    unittest.main()
