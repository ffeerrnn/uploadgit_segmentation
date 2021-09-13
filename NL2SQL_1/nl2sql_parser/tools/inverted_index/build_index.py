# -*- coding: utf-8 -*-

import jieba
import traceback
import logging

from nl2sql_parser.tools.data_manage.table_info_loader import TableInfoLoader

class BuildIndex(object):

    def __init__(self):

        self.inverted_index = {}
        self.data_loader = TableInfoLoader()
        self.logger = logging.getLogger('develop')

    def init(self):
        """
        初始化的时候对表名中的词和字段中的词建立倒排索引
        :return:
        """
        table_ids = self.data_loader.get_table_ids_from_db()

        for table_id in table_ids:
            try:
                meta = self.data_loader.get_table_info(table_id)
            except BaseException:
                self.logger.info("Read data from meta failed.")
                self.logger.debug(traceback.format_exc())
                continue

            # 对字段中的字和表名中的字建立倒排索引
            self._build_index(table_id, meta)

    def _build_index(self, table_id, meta):

        if meta and "fields" in meta:
            for field in meta["fields"]:
                words = list(filter(lambda w: len(w) > 1, jieba.cut(field["alias"])))
                for word in words:
                    characters = list(word)
                    for character in characters:
                        if character != " " and character:
                            self.inverted_index.setdefault(character.lower(), set())
                            self.inverted_index[character.lower()].add(table_id)

                if field["type"] == "DATE":
                    self.inverted_index.setdefault("${DATE}", set())
                    self.inverted_index["${DATE}"].add(table_id)

        if meta and 'name' in meta:
            for name in meta['name']:
                words = list(filter(lambda w: len(w) > 1, jieba.cut(name)))
                for word in words:
                    characters = list(word)
                    for character in characters:
                        if character != " " and character:
                            self.inverted_index.setdefault(character.lower(), set())
                            self.inverted_index[character.lower()].add(table_id)

        if meta and 'type_name' in meta:
            for name in meta['type_name']:
                words = list(filter(lambda w: len(w) > 1, jieba.cut(name)))
                for word in words:
                    characters = list(word)
                    for character in characters:
                        if character != " " and character:
                            self.inverted_index.setdefault(character.lower(), set())
                            self.inverted_index[character.lower()].add(table_id)

build_index = BuildIndex()

if __name__ == '__main__':

    from django.conf import settings
    import NL2SQL.settings as app_setting
    settings.configure(default_settings=app_setting)

    build_index.init()
    print(build_index.inverted_index)