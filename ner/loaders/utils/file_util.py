from typing import Iterator
from itertools import (takewhile, repeat)


class FileUtil():

    @staticmethod
    def line_iter(filename: str) -> Iterator[str]:
        """line iterator

        Args:
            filename (str): opening file

        Yields:
            Iterator[str]: file line iterator
        """
        with open(filename, 'r', encoding='utf-8') as f:
            line = f.readline()
            while line:
                yield line
                line = f.readline()

    @staticmethod
    def count_lines(filename: str) -> int:
        """get lines count

        Args:
            filename (str): opening file

        Returns:
            int: count lines
        """
        buffer = 1024*1024
        with open(filename, "r", encoding='utf-8') as f:
            buf_gen = takewhile(lambda x: x, (f.read(buffer)
                                for _ in repeat(None)))
            return sum(buf.count('\n') for buf in buf_gen)
