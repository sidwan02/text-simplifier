def test(t):
    return t

class ClassTest(object):
    def test_def(self):
        test_msg = test('Hi')
        print(test_msg)

# # Creates new instance.
# my_new_instance = ClassTest()
# # Calls its attribute.
# my_new_instance.test_def()