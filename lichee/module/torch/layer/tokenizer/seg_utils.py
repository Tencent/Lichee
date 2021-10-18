# -*- coding: utf-8 -*-
# auth:lshzhang
# date:2010.09.03
import time
# import os
# import sys


# seg_mode = TC_CRF | TC_POS | TC_CUS | TC_CUT | TC_ADD | TC_ASCII_LM | TC_ASCII_LM_PINYIN
# seg_mode = TC_NER_DL | TC_CRF | TC_POS | TC_CUS | TC_CUT | TC_ADD | TC_ASCII_LM |
#            TC_ASCII_LM_PINYIN | TC_VIDEO | TC_OTHER_NE


class SegUtils:
    noun_pos = set(['n', 'ng', 'nr', 'ns', 'nt', 'nx', 'nz', 'vn'])
    loc_pos = set(['ns'])
    per_pos = set(['nr'])
    org_pos = set(['nt'])

    def __init__(self, qqseg_path, init_mode=0):
        if qqseg_path != None and len(qqseg_path) > 0:
            # add include
            sys.path.append(qqseg_path + "/include")
            # add lib64
            sys.path.append(qqseg_path + "/lib64")
            from segmentor_4_python3 import TC_PER_W, TC_NER_DL, TC_POS, TC_CUS, TC_CUT, TC_ADD, \
                TC_ASCII_LM, TC_ASCII_LM_PINYIN, TCInitSeg, TCAddCustomDict, TCSwitchCustomDictTo, \
                TCCreateSegHandle, TCSegment, TC_UTF8, TCGetMixWordCnt, TCGetMixWordAt

        stime = time.time()
        qqsegconf = qqseg_path + "/data/"
        if init_mode == 0:
            self.seg_mode = TC_PER_W | TC_NER_DL | TC_POS | TC_CUS | TC_CUT | TC_ADD | TC_ASCII_LM | TC_ASCII_LM_PINYIN
            if not TCInitSeg(qqsegconf, True):
                print("Fail to init seg")
                exit(0)
        else:
            self.seg_mode = TC_POS | TC_CUS | TC_CUT | TC_ADD | TC_ASCII_LM | TC_ASCII_LM_PINYIN
            if not TCInitSeg(qqsegconf, False):
                print("Fail to init seg")
                exit(0)
        print("Init seg ok, cost ", time.time() - stime)

        # 加载自定义词典
        dict_id = TCAddCustomDict('custom_seg_dict/GBK/')
        TCSwitchCustomDictTo(dict_id)

        self.handle = None

    def init_handle(self):
        if self.handle is not None:
            return

        from segmentor_4_python3 import TC_PER_W, TC_NER_DL, TC_POS, TC_CUS, TC_CUT, TC_ADD, \
            TC_ASCII_LM, TC_ASCII_LM_PINYIN, TCInitSeg, TCAddCustomDict, TCSwitchCustomDictTo, \
            TCCreateSegHandle, TCSegment, TC_UTF8, TCGetMixWordCnt, TCGetMixWordAt

        stime = time.time()
        # handle
        self.handle = TCCreateSegHandle(self.seg_mode)
        print('init handle, cost ', time.time() - stime)

    # output mix seg result
    def mix_seg(cls, text):
        cls.init_handle()

        from segmentor_4_python3 import TCSegment, TC_UTF8, TCGetMixWordCnt, TCGetMixWordAt

        if not TCSegment(cls.handle, text, len(text.encode('utf-8')), TC_UTF8):
            print("Segment sentence failed!")
            return None
        res_mix_count = TCGetMixWordCnt(cls.handle)
        seg_res = []
        for i in range(res_mix_count):
            word = TCGetMixWordAt(cls.handle, i)
            seg_res.append(word)
        return seg_res

    # output mix seg result with pos
    def mix_seg_with_pos(cls, text):
        cls.init_handle()

        from segmentor_4_python3 import TCSegment, TC_UTF8, TCGetMixWordCnt, TCGetMixWordAt, TCGetMixTokenAt, \
            TCPosId2Str

        if not TCSegment(cls.handle, text, len(text.encode('utf-8')), TC_UTF8):
            print("Segment sentence failed!")
            return None
        res_mix_count = TCGetMixWordCnt(cls.handle)
        seg_res = []
        for i in range(res_mix_count):
            word = TCGetMixWordAt(cls.handle, i)
            pos_id = int(TCGetMixTokenAt(cls.handle, i).pos)
            pos_str = TCPosId2Str(pos_id)
            seg_res.append((word, pos_str))
        return seg_res

    # output sub seg result with pos
    def basic_seg(cls, text):
        cls.init_handle()

        from segmentor_4_python3 import TCSegment, TC_UTF8, TCGetResultCnt, TCGetWordAt

        if not TCSegment(cls.handle, text, len(text.encode('utf-8')), TC_UTF8):
            print("Segment sentence failed!")
            return None
        res_mix_count = TCGetResultCnt(cls.handle)
        seg_res = []
        for i in range(res_mix_count):
            word = TCGetWordAt(cls.handle, i)
            seg_res.append(word)
        return seg_res

    # output sub seg result with pos
    def basic_seg_with_pos(cls, text):
        cls.init_handle()

        from segmentor_4_python3 import TCSegment, TC_UTF8, TCGetResultCnt, TCGetWordAt, TCGetAt, TCPosId2Str

        if not TCSegment(cls.handle, text, len(text.encode('utf-8')), TC_UTF8):
            print("Segment sentence failed!")
            return None
        res_mix_count = TCGetResultCnt(cls.handle)
        seg_res = []
        for i in range(res_mix_count):
            word = TCGetWordAt(cls.handle, i)
            pos_id = int(TCGetAt(cls.handle, i).pos)
            pos_str = TCPosId2Str(pos_id)
            seg_res.append((word, pos_str))
        return seg_res

    # output phrase seg result,  after word seg
    def phrase_seg(cls):
        cls.init_handle()

        from segmentor_4_python3 import TCGetPhraseCnt, PHRASE_NAME_IDX, PHRASE_NAME_FR_IDX, TCGetPhraseAt, \
            TCGetPhraseTokenAt, TCPosId2Str, PHRASE_LOCATION_IDX, PHRASE_ORGANIZATION_IDX

        stime = time.time()
        res_phrase_count = TCGetPhraseCnt(cls.handle)
        seg_res = []
        ner_res = []
        NAME_IDX = set([PHRASE_NAME_IDX, PHRASE_NAME_FR_IDX])

        for i in range(res_phrase_count):
            word = TCGetPhraseAt(cls.handle, i)
            pos_id = int(TCGetPhraseTokenAt(cls.handle, i).pos)
            pos_str = TCPosId2Str(pos_id)
            id_cls = int(TCGetPhraseTokenAt(cls.handle, i).cls)
            if id_cls in NAME_IDX:
                ner_res.append((word, "PER"))
            if id_cls == PHRASE_LOCATION_IDX:
                ner_res.append((word, "LOC"))
            if id_cls == PHRASE_ORGANIZATION_IDX:
                ner_res.append((word, "ORG"))
            seg_res.append((word, pos_str))
        return seg_res, ner_res

    def n_gram(cls, words, max_n=1):
        ngs = []
        for i in range(max_n):
            ngs.append([])

        w_len = len(words)
        i = 0
        while i < w_len:
            for j in range(max_n):
                if i + j < w_len:
                    # print i, j, ''.join(words[i:i+j+1])
                    ngs[j].append(''.join(words[i:i + j + 1]))
            i += 1
        return ngs

    def destroy_handle(cls):
        from segmentor_4_python3 import TCCloseSegHandle, TCUnInitSeg

        # un-initialization
        if cls.handle is not None:
            TCCloseSegHandle(cls.handle)
        TCUnInitSeg()


if __name__ == "__main__":
    seg_utils = SegUtils()

    # test
    texts = []
    text = '长发美女穿着球衣加黑色短裤跳舞好迷人'
    texts.append(text)
    text = '长发美女穿红色连体裙跳舞好迷人'
    texts.append(text)
    text = '长发美女穿着短裙在家跳舞好迷人'
    texts.append(text)
    text = '我爱北京天安门'
    texts.append(text)
    text = '少年派：邓小琪花痴着钱三一，林妙妙在一旁吐槽他'
    texts.append(text)
    # for text in texts:
    import sys

    text = sys.argv[1]
    print('raw:', text)

    stime = time.time()
    res = seg_utils.mix_seg(text)
    print('mix seg:', ' '.join(res))

    basic_res = seg_utils.basic_seg(text)
    print('basic seg:', ' '.join(basic_res))

    res = seg_utils.mix_seg_with_pos(text)
    print('mix seg with pos:', )
    for w, p in res:
        print(w, p, end=' ')
    print('')

    res = seg_utils.basic_seg_with_pos(text)
    for w, p in res:
        print(w, p, end=' ')
    print('')

    phrase_res = seg_utils.phrase_seg()
    print(res)

    for w, p in phrase_res[0]:
        print(w, p, end=' ')
    print('')

    for w, p in phrase_res[1]:
        print(w, p, end=' ')
    print('')
    print(time.time() - stime)
