# -*- coding: utf-8 -*-

import jieba.posseg as pseg
from nl2sql_parser.tools.data_manage.place_name_dataloader import PlaceNameDataloader

data_loader = PlaceNameDataloader()

class PlaceNameExtractor(object):
    """
    行政区划查询类
    """
    def __init__(self):
        self.keywordprocessor = data_loader.get_keywordprocessor()
        self.name_full_name_dict = data_loader.get_cache('name_full_name_dict')
        self.full_name_name_dict = data_loader.get_cache('full_name_name_dict')

    def pop_upper_place(self, new_place, place_list, full_name_place_list):
        """
        如果提取到了某地区的上级地区，将上级地区删除
        :param ad:
        :param ad_list:
        :return:
        """
        if not place_list:
            place_list.append(new_place)
            full_name_place_list.append(self.name_full_name_dict[new_place][0]) # 一个检测对应多个地方时，先取第一个
            
        else:
            # 如果新地名的全称包含了前一个地名全称中的任何一个，则删除前一个地名
            last_place = place_list[-1]
            last_place_full_name_list = self.name_full_name_dict[last_place]
            new_place_full_name_list = self.name_full_name_dict[new_place]
            drop_upper = False
            for name_i in last_place_full_name_list:
                for name_j in new_place_full_name_list:
                    if name_i in name_j:
                        drop_upper = True
                        break
                if drop_upper == True:
                    place_list[-1] = new_place
                    full_name_place_list[-1] = name_j
                    break

        return place_list, full_name_place_list        

    def extract(self, text):
        match_results = self.keywordprocessor.extract_keywords_nlp(text, span_info=True) # 硬匹配到的地区名称
        words = pseg.lcut(text)
        word_pos_dict = {}
        for w in words:
            if w.word in word_pos_dict:
                word_pos_dict[w.word].append(w.flag)
            else:
                word_pos_dict[w.word] = [w.flag]
        new_match_results = []
        for result in match_results:
            if len(result[0]) >= 2 or (result[0] in word_pos_dict and 'ns' in word_pos_dict[result[0]]):
                new_match_results.append(result[0])

        place_list = []
        full_name_place_list = []
        for new_place in new_match_results:
            place_list, full_name_place_list = self.pop_upper_place(new_place, place_list, full_name_place_list)

        place_list = [self.full_name_name_dict[i] for i in full_name_place_list]
        return place_list

if __name__ == '__main__':
    
    ad_seacher = PlaceNameExtractor()
    text = '石家庄新华'
    result = ad_seacher.extract(text)

    print(result)
