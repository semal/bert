from bert_serving.client import BertClient
from scipy.spatial.distance import cosine


def test():
    bc = BertClient(ip='10.0.5.133')
    test_intros = [
        '引力传媒成立于2002年，是一家整合营销传播服务商，2015年5月在上交所上市。',
        '山东闻道通信技术有限公司是一家商用智能wifi营销方案提供商，致力于打造无线网络的最优平台，为社会提供最先进和最优质的无线网络服务，主要产品有wifi广告营销平台、app快分发wifi营销平台和微信粉丝通wifi营销平台。',
        '饭美美是一家互联网外卖盒饭订餐服务商，用户可通过自助售饭机“饭饱宝”订餐、手机app在线订餐、微信外卖配送订餐等方式获取安全、新鲜、健康、便捷的在线外卖盒饭订餐服务。',
        '冠游时空是一个互联网游戏及应用的开发者平台，关注actionscript，flex，移动开发，游戏开发以及unity3d等各类新兴ria技术，致力于帮助解决ria开发者在开发过程中遇到的各种问题，指引并协助开发者获得更好的职业发展空间。'
    ]
    res = bc.encode(test_intros)
    if res is not None:
        print(cosine(res[0], res[1]))
        print(cosine(res[0], res[2]))


if __name__ == '__main__':
    test()
