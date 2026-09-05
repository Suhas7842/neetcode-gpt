from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        result = []

        for number in numbers:
            text = str(number)
            tokens = []
            i = 0

            while i < len(text):
                # Find the longest matching token
                match = None

                for token in vocab:
                    if text.startswith(token, i):
                        if match is None or len(token) > len(match):
                            match = token

                if match:
                    tokens.append(match)
                    i += len(match)
                else:
                    # Move forward if no token matches
                    tokens.append(text[i])
                    i += 1

            result.append(tokens)

        return result

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        i = 0
        count = 0

        while i < len(text):
            match = None

            for token in vocab:
                if text.startswith(token, i):
                    if match is None or len(token) > len(match):
                        match = token

            if match:
                count += 1
                i += len(match)
            else:
                count += 1
                i += 1

        return count

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        tokens = self.count_tokens(text, vocab)

        # Words separated by whitespace
        words = len(text.split())

        return round(tokens / words, 4)
