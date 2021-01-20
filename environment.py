

class Environment:
    """ An interface for environments. """

    def current_state(self):
        # Метод должен вернуть определенную структуру данных:
        # user: {id: int, features: vect}
        # docs: list [{id: int, features: vect}]
        raise NotImplementedError

    def reaction(self, user_id, doc_id):
        # Метод должен вернуть какое-то число
        raise NotImplementedError


class ConstEnvironment(Environment):
    def __init__(self, users, documents, reactions):
        self.users = users
        self.documents = documents
        self.n = len(users)
        self.reactions = reactions
        self.__current_state_gen = self.__current_state()

    def __current_state(self):
        i = 0
        while True:
            i = i % self.n
            yield (self.users[i], self.documents)
            i += 1

    def current_state(self):
        # TODO надо переделать слишком мудрено
        return next(self.__current_state_gen)

    def reaction(self, user_id, doc_id):
        return self.reactions[user_id]
