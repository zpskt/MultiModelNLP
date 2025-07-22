

if __name__ == '__main__':
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    semantic_cls = pipeline(Tasks.text_classification, 'iic/nlp_structbert_sentiment-classification_chinese-large')
    pre_list = ['启动的时候很大声音，然后就会听到1.2秒的卡察的声音，类似齿轮摩擦的声音',
                '我很喜欢',
                '吴总好帅',
                '海尔欠我的工钱到底什么时候才给我',
    ]
    pre_list_2 = ['启动的时候很大声音，然后就会听到1.2秒的卡察的声音，类似齿轮摩擦的声音'] * 100
    cls = semantic_cls(input=pre_list_2)

    print(cls)


