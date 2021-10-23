from typing import *


class TagConvert():
    def __init__(self, rules: Dict[str, str], default_tag: str = 'O'):
        self.rules: Dict[str, str] = rules
        self.default_tag: str = default_tag

    def tag2prompt(self, tag: List[str], word: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """ convert tag to a prompt

        Args:
            tag (List[str]): tag list. e.g. ['B-position','I-position']
            word (List[str]): character list. e.g. ['教','师']

        Raises:
            ValueError: the length of word is not equal to the tag
            KeyError: tag not found in rules

        Returns:
            List[str]: prompt. e.g. ['教','师','是','一','个','职','位','[SEP]']
            List[str]: prompt_mask e.g. ['0','0','1','1','1','0','0','1']
            List[str]: prompt_tags e.g. ['B-position','I-position','O','O','O','O','O','O']
        """
        if len(word) != len(tag):
            raise ValueError(f"the length of word is not equal to the tag")
        # tag[0] for example: B-position, get the tag: position
        single_tag = tag[0].split('-')[-1]
        if single_tag not in self.rules:
            raise KeyError(f"tag: {single_tag} not found in rules")
        prompt = list(f"{''.join(word)}是一个{self.rules[single_tag]}")+['[SEP]']
        prompt_mask = [0 for _ in word] + \
            ([1]*3) + [0 for _ in self.rules[single_tag]] + [1]
        prompt_tags = tag + \
            [self.default_tag for _ in range(len(word), len(prompt))]
        assert len(prompt) == len(prompt_mask) and len(
            prompt_mask) == len(prompt_tags)
        return prompt, prompt_mask, prompt_tags
