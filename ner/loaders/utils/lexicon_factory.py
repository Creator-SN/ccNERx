from typing import List
from ner.loaders.utils import *
from tqdm import *
import json


class TrieFactory():

    @staticmethod
    def get_trie_from_vocabs(vocab_files: List[str], max_line: int = None) -> Trie:
        """build lexicon trie from vocab file

        Args:
            vocab_files (List[str]): list of files
            max_line (int, optional): maximum line. Defaults to None.

        Returns:
            Trie: builded trie
        """
        vocabs = set()
        for file in vocab_files:
            file_lines = FileUtil.count_lines(file)
            for index, line in tqdm(enumerate(FileUtil.line_iter(file)), desc="load vocabs into trie", total=file_lines):
                if max_line >= 0 and index >= max_line:
                    break
                line = line.strip().split()
                word = line[0].strip()
                vocabs.add(word)
        lexicon_tree: Trie = Trie()
        for word in vocabs:
            lexicon_tree.insert(word)
        return lexicon_tree

    def get_all_matched_word_from_dataset(dataset_files: List[str], lexicon_tree: Trie) -> List[str]:
        """Get All Matched word from dataset json file

        Args:
            dataset_files (List[str]): dataset json file path, a json object each line
            lexicon_tree (Trie): lexicon trie

        Returns:
            List[str]: all matched words are sorted in order
        """
        matched_words = set()
        for file in dataset_files:
            file_lines = FileUtil().count_lines(file)
            for line in tqdm(FileUtil.line_iter(file), desc="load dataset matched word", total=file_lines):
                data = json.loads(line.strip())
                assert 'text' in data, 'dataset type error, expected text property in object'
                text = data['text']
                sent = [ch for ch in text]
                for word in lexicon_tree.getAllMatchedWords(sent):
                    matched_words.add(word)
        return sorted(matched_words)

    
