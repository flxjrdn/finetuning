import unittest

from generation import AnswerGenerator, PROMPT


class MyTestCase(unittest.TestCase):
    def test_create_prompt(self):
        prompt = AnswerGenerator.create_prompt(
            question="test question",
            chunks=["chunk 1", "chunk 2"]
        )
        self.assertTrue("test question" in prompt)
        self.assertTrue("chunk 1\nchunk 2" in prompt)
        self.assertTrue(prompt.startswith(PROMPT[:20]))
        self.assertFalse("PLACEHOLDER" in prompt)


if __name__ == '__main__':
    unittest.main()
