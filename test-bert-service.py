from bert_serving.client import BertClient
from scipy.spatial.distance import cosine


def test():
    bc = BertClient(ip='10.0.5.133')
    test_intros = [
        '罗永浩卸任CEO？对不起，没有罗的锤子，活不过一个月！ ',
        '锤子真的要凉？传罗永浩卸任CEO，并开始遣散各地员工',
        '微信最强的对手不是子弹短信，而是暗自发展的米聊！',
        '麒麟980+4800万主摄配TOF镜头，荣耀V20正式发布售价2999元起',
        '今日头条欲收购锤子科技部分专利使用权 锤子手机为何走不下去？',
        '今日头条欲购 锤子科技部分专利使用权 锤子手机为何走不下去？ | 凤凰网',
    ]
    res = bc.encode(test_intros)
    if res is not None:
        print(1 - cosine(res[0], res[1]))
        print(1 - cosine(res[0], res[2]))
        print(1 - cosine(res[0], res[3]))
        print(1 - cosine(res[4], res[5]))


if __name__ == '__main__':
    test()
