# 需要版本3.8

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

#初始化纠错pipeline
model_id = 'iic/nlp_bart_text-error-correction_chinese'
pipeline = pipeline(Tasks.text_error_correction, model=model_id, model_revision='v1.0.1')

#单条调用
input = '这洋的话，下一年的福气来到自己身上。'
result = pipeline(input)
print(result['output'])


#批量调用
inputs = ['这洋的话，下一年的福气来到自己身上。', '在拥挤时间，为了让人们尊守交通规律，派至少两个警察或者交通管理者。', '随着中国经济突飞猛近，建造工业与日俱增']
batch_out = pipeline(inputs, batch_size=2)
for result in batch_out:
    print(result['output'])
