import re
import torch
import torch.nn as nn
import numpy as np
from Sentiment.pytorch_pretrained import BertModel, BertTokenizer

import sys

sys.path.append('Sentiment.zip')


class Config(object):
    """配置参数"""

    def __init__(self):
        self.model_name = 'bert'
        self.class_list = ['中性', '积极', '消极']  # 类别名单
        self.save_path = './Sentiment/Sentiment/saved_dict/bert.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 3  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = './Sentiment/bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out


PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def clean(text):
    # text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
    # text = re.sub(r"\[\S+\]", "", text)  # 去除表情符号
    # text = re.sub(r"#\S+#", "", text)  # 保留话题内容
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text = re.sub(URL_REGEX, "", text)  # 去除网址
    text = text.replace("转发微博", "")  # 去除无意义的词语
    text = re.sub(r"\s+", " ", text)  # 合并正文中过多的空格
    return text.strip()


def load_dataset(data, config):
    pad_size = config.pad_size
    contents = []
    for line in data:
        lin = clean(line)
        token = config.tokenizer.tokenize(lin)  # 分词
        token = [CLS] + token  # 句首加入CLS
        seq_len = len(token)
        mask = []
        token_ids = config.tokenizer.convert_tokens_to_ids(token)

        if pad_size:
            if len(token) < pad_size:
                mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                token_ids += ([0] * (pad_size - len(token)))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size
        contents.append((token_ids, int(0), seq_len, mask))
    return contents


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches  # data
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):  # 返回下一个迭代器对象，必须控制结束条件
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):  # 返回一个特殊的迭代器对象，这个迭代器对象实现了 __next__() 方法并通过 StopIteration 异常标识迭代的完成。
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, 1, config.device)
    return iter


def match_label(pred, config):
    label_list = config.class_list
    return label_list[pred]


def final_predict(config, model, data_iter):
    map_location = lambda storage, loc: storage
    model.load_state_dict(torch.load(config.save_path, map_location=map_location))
    model.eval()
    predict_all = np.array([])
    with torch.no_grad():
        for texts, _ in data_iter:
            outputs = model(texts)
            pred = torch.max(outputs.data, 1)[1].cpu().numpy()
            pred_label = [match_label(i, config) for i in pred]
            predict_all = np.append(predict_all, pred_label)

    return predict_all


def predict(text):
    config = Config()
    model = Model(config).to(config.device)
    test_data = load_dataset(text, config)
    test_iter = build_iterator(test_data, config)
    result = final_predict(config, model, test_iter)
    for i, j in enumerate(result):
        print('text:{}'.format(text[i]))
        print('label:{}'.format(j))


if __name__ == '__main__':
    test = [
        '我带孩子在天坛的详细路线和时间安排,早上起床看阳光明媚,微风不冷马上线上小程序购票预约,周一祈年殿回音壁都不开,所以票非常充裕~随机美团外卖,下单了核桃、山核桃、花生、瓜子.1 吃完早饭,坚果正好送到(请大家买那些原味的没有盐和其它味道的坚果)下楼做了核酸,不紧不慢的出发了;2 十点半到达天坛公园东门停车场回,车位非常充足,大家放心开车前往~;3 东门检票进去就是咖啡厅,买了招牌热巧和闪光棒棒糖,热巧味道不错,棒棒糖绿色的是哈密瓜味儿~这里还有纪念品~;4一路向西门走,就来到了祈年殿,这时差不多十一点了,周一闭馆,只能在外面拍拍照片,其实带孩子,进不进;去差不多~;5 继续向西,穿过牡丹园,路过一个小亭子,前面有一颗贼漂亮的银杏树,拍照逗留了十分钟照片张张可保留~;6 继续向西,开始出现没有栅栏的小树林了~准备开始敲核桃吧~需要点儿耐心,松鼠们警惕性也挺高的,得过了十分钟,开始有从树上下来的了,陆陆续续从灌木丛里、其它书上跳来了七八只~看着我儿子蹑手蹑脚、嘴巴里还细声细语的说:小松鼠,我来给你们送好吃的啦~好可爱~;7快十二点时,我们继续往西走,有橙色的枫叶、金黄的银杏、绿油油的松柏,也留下了几张美照~;8 也就一百多米,就又是一片松树林,这次是一只胆大的长尾巴松鼠都直接就跑到我儿子面前~叼着核桃就跑去藏,然后又跑回来、跑去藏一特别有趣~;9 这里就走到头了,稍微往南一点点就是门,但这不是西门的大门,是第二道,从这出去,右手边就是天坛新来的那家;餐厅,另一篇有写笔记~大约12:45我们进入了餐厅开始吃饭~10 吃完饭,休息了一下,快两点了,我们原路返回~孩子在家一般是一点多睡觉,但是要想在车里顺利睡着,需要耗到三点多~于是我们又在东门的咖啡厅里分享了一个小兔子慕斯~孩子觉得这次逛园子非常完美哈哈~临走还买了一个盲盒,是榫卯结构的小老虎,抽中了喜欢的绿色,很开心~东门检票口又卫生间,孩子上车前哗哗一下,上车还没开到收费亭就秒睡了~呼吸新鲜空气、跟小松鼠近距离接触、买了小玩具、尝了尝棒棒糖。拍了好多照片,这次出行,完美~下次睡醒午觉了再去一次,看看开灯了的天坛~',
        '下午和老妈带着孩子来天坛公园玩儿,南门进的,由于快三点了阳光有好多地方都照不到,总觉得阴冷阴冷的,尤其是走,到斋宫附近,总感觉这里的树都与众不同(可能是古树的缘故)!而且傍晚有很多乌鸦盘旋上空,停留在斋宫的屋顶,看着更感觉阴森了!从小家住先农坛离西门近,小学更是西门旁边的紫竹林,小学田径锻炼也是西门近左手边的小道道,小时候最喜欢的就是西门近直走也就五百米左右的龙宫,对面的游乐场也是儿时的回忆!几十年不曾来过天坛了,再来就物是人非了!商业模式已开启二道门附近的咖啡馆进去点了一杯拿铁,一杯巧克力,一个冰淇淋球一共102元,真是哎不说了!开心就好!现在淡季门票10元,像神乐坊,圆丘,祈年殿,回音壁等都没有开放!',
        '天坛公园位于北京城南部,是国家AAAAA级景区,是古代皇帝祭祀黄天,祈祷五谷丰登的地方。这个季节来天坛,正是银杏叶子变黄的时候。东门进入公园位于东北部有大片银杏树可供欣赏。美丽的叶子在蓝天阳光下无比动人。沿着小道前行,秋季的美让人词穷,唯有用心感受。在西南部的斋宫近期可申请免费预约入内参观。有时间可以去看看,人不多,此处清静而美好。',
        '看了一眼大众点评,我居然是2012年12月4日来的天坛,这马上就要满10周年了,这次是刷到某书上这个时节正好是松鼠屯粮的时候,只要带好核酸,就能偶遇松鼠,我们去的时候没有找到,去看了祈年殿,确实很壮观,由28根金丝楠木大柱支撑,柱子环转排列,中间4根“龙井柱”,支撑上层屋檐;中间12根金柱支撑第二层屋檐,外围12根檐柱支撑第三层屋檐;相应设置三层天花板,中间设置龙凤藻井;殿内梁枋施龙凤和玺彩画。祈年殿中间4根“龙井柱”,象征着一年的春夏秋冬四季;中层十二根大柱比龙井柱略细,名为金柱,象征一年的12个月;外层12根柱子叫檐柱,象征一天的12个时辰。中外两层柱子共24根,象征24节气,等后面我们准备走的时候,很多孩子拿着核桃敲击在地上发出声音,召唤了很多小松鼠,这的小松鼠都不怕人,是从我手里拿走的核桃,希望大家喂食的时候,一定不要给松鼠喂一些添加味道的东西哈。',
        '关于,一些岁月静好。赶上秋天的小尾巴和邻居来逛天坛公园。我俩都盘了下,似乎都有很多年没来过天坛了,我有印象的那次还是高考毕业后.天坛还是儿时候的样子,没怎么变,皇家祭祀场所,宁谧肃静,神圣不可侵犯。基础设施修建的更好了,老少皆宜,非常适合来逛园子。关于买票:手机扫码当天预约门票即可,没有月卡的成人28元的套票!可以去祈年殿、九龙壁、圆丘。值得入手~二维码扫码或者身份证刷卡入园即可!非常方便~关于值得入手的打卡必备冰:我们从西门进来,走个大概10分钟就能看到新晋的打卡地,有休息的餐厅,卖天坛主题冰淇凌和各种奶茶咖啡,我点了他家夏季版的水果茶,天气太冷啦,要的常温~也还好。雪顶是奶油、点缀的花花是巧克力,天坛的造型是冰淇淋!价格小贵,但用材很不错!可以说,物超所值!拿着她逛园子,我就是公园里最靓的女仔!关于拍照:祈年殿必须是要好好拍的!但赶上周末人实在是太多了,准备冬天的时候再来好好拍照!不过,也有意外收获的,因为天气很好,夕阳洒在汉白玉的栏杆上,映出金灿灿的光,护栏上雕刻的祥云和飞龙被映衬的更加栩栩如生!更显得祈年殿的皇家贵气!好啦~就安利到这里!准备明年办一张公园年票,时不时的要来这边转转呐!就酱~',
        '听说天坛公园有菊花展,我和朋友一起相约来到公园看展菊,展菊是在天坛公园祈年殿里,对于我们这些60岁以上的老人进入祈年殿都是免费的。菊花主要是盆景展示,各种五颜六色绚丽多彩。菊花各有特色,有的秀丽淡雅,有的鲜艳夺目,有的昂首挺胸菊花傲霜怒放,五彩缤纷,千姿百态。白的似雪,粉的似霞,大的像团团彩球,小的像盖盖精巧的花灯。那一团团一簇簇的菊花,正在拔态怒放。时不时还闻到菊花淡淡的香气。一次新的赏花体验。非常的开心。',
        '出行:地铁7号线,桥湾!东南口出,步行800米左右即可到达天坛公园.先扫码,大门左边窗口上贴着线上购买门票的二维码,普通门店14元,带祈年殿门票34元.上周开了菊花花卉展,进入公园,还可以看到不同品类的菊花哦夏菇凉在慢慢退场..秋菇凉已经迫不及待登场啦......北京城的秋天虽然短暂风景也是最美的....六百年风雨,三千年风华,天坛公园带你领略中国历史文化',
        '#你好2020#新年第一天元气满满的早起出门买早饭结果高估了自己抗冻能力回家成功冻发烧（大概是想告诉我2020要量力而行）然鹅这并不影响后续计划一出门立马生龙活虎新年和新??更配哦??看了误杀吃了大餐就让新的一年一直这样美滋滋下去吧??',
        '大宝又感冒鼻塞咳嗽了，还有发烧。队友加班几天不回。感觉自己的情绪在家已然是随时引爆的状态。情绪一上来，容易对孩子说出自己都想不到的话来……2020年，真的要学会控制情绪，管理好家人健康。这是今年最大的目标。?',
        '还要去输两天液，这天也太容易感冒发烧了，一定要多喝热水啊?',
        '我太难了别人怎么发烧都没事就我一检查甲型流感?',
        '果然是要病一场的喽回来第三天开始感冒今儿还发烧了喉咙眼睛都难受的一匹怎么样能不经意让我的毕设导师看到这条微博并给我放一天假呢?',
        '听说天坛公园有菊花展,我和朋友一起相约来到公园看展菊，展菊是在天坛公园祈年殿里，对于我们这些60岁以上的老人.进入祈年殿都是免费的。菊花主要是盆景展示,各种五颜六色绚丽多彩。菊花各有特色,有的秀丽淡雅，有的鲜艳夺目,有的昂首挺胸菊花傲霜怒放,五彩缤纷,千姿百态。白的似雪,粉的似霞,大的像团团彩球,小的像盖盖精巧的花灯。那一团团一簇簇的菊花,正在拔蕊怒放。时不时还闻到菊花淡淡的香气。一次新的赏花体验。非常的开心。',
        '出行:地铁7号线,桥湾!东南口出,步行800米左右即可到达天坛公园,先扫码,大门左边窗口上贴着线上购买门票的二维码,普通门店14元,带祈年殿门票34元,上周开了菊花花齐展,进入公园,还可以看到不同品类的菊花哦,夏菇凉在慢慢退场,秋廷凉已经迫不及待登场啦。...北京城的秋天虽然短暂风景也是最美的.六百年风雨,三千年风华,天坛公园带你领略中国历史文化',
        '中国北京天坛是中国文化遗产,中国十大遗址,北京最受欢迎景点之一。是明清皇帝祭祀用的场所,主要有两大活动:祭天、祈谷。逛天坛我觉得必看的应该是祈年殿,圆丘坛和丹陛桥,从观感而言这几个地方是打卡的好地方,其他的地方就,听导游讲解就OK了,有一些个地方还是要回避一下为好。',
        '具体在哪个位置不知道外的人来办事儿就上这儿溜溜达,4月份去的,其实花啥还没开,树叶倒是绿了,感觉其实应该夏,天来特别棒,确实挺美的,配上蓝天,我去的时候是工作日没什么人溜达,一大圈得走半天,园子里有吃饭的地方不算便宜,各种网红冰淇淋,总之推荐来走一走,时间比较闲走一走挺好。',
        '听说天坛公园里有很多可爱的小松鼠,今天就专程为这些小家伙儿送去些冬储粮,进公园北门没多远就有一片安静的松树林,站在林间小路仔细观瞧,果然看到四、五只松鼠在树上来回跳跃。他们身型小巧,毛皮呈棕红色,圆圆的小脑袋上顶着一对尖尖的耳朵、一双黑溜溜的小眼睛特别机灵,蓬松的大尾巴是它们的特有的标志。扔一颗花生过去,它们很快就从树上跳下来,跑过去用小爪子抓起花生送进口中,津津有味的大吃起来,那样子真是太萌了。有时候它们放进嘴里并不吃,而是用爪子在地上创个小坑,然后把花生放进去埋起来,为即将到来的冬天做储备。赶上胆子大些的小家伙也会跑过来把你放在手心里的花生拿走,非常有趣。听工作人员说公园里还有魔王松鼠,就是体型偏大,耳朵里长着长长的毛的那种,但我今天没有看到,以后有机会再过来慢慢寻觅吧。',
        '18元的棒棒糖,里面居然是夹纸(所谓的精美图案),没注意孩子已经把纸吃了。什么玩意儿',
        '疫情防控下还扎堆征婚居然还有不带口罩的,真是什么也阻挡不了他(她)们征婚,不过肯定没有什么能成的,因为人太多一山望着一山高,都想找个条件更好的,都是一些老油条了,平时有很多跑步运动的。公园绿化挺好很多明清时候的古树。',
        '门口租讲解器那女的态度十分恶劣,真给服务行业丢脸,不想租还不让退,一定要用现金租。死脑筋子,我祝他一辈子也挣不了钱。都不想逛天坛了,影响心情。劝大家别租讲解器,网上找个讲解就行别花冤枉钱。栓Q了。',
        '我是不理解这样子的景区是怎么定义为5a的?凭的什么,没劲,简直浪费钞票好吧',
    ]
    predict(test)
