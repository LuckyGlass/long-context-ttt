## 通用参数
### `ModelArguments`
模型的参数。

- `model_name_or_path`
- `model_max_length`：这个参数会决定输入到模型的token数（如果`pad_to_max_length=True`）。
- `is_peft_model`：`model_name_or_path` 是否指向一个peft（lora）model。

### `DataTrainingArguments`
通用数据参数，用于从单条长文本生成 `ContextDataset`以及生成测试模板。

- `block_size`
- `len_segment`
- `len_offset`
- `num_generate_qa`：生成通用synthetic QA的数量，设置为0时即不引入通用synthetic QA。如果设置为非零值，则要求指定 `generator_name_or_path`。
- `generator_name_or_path`：用于生成synthetic QA的模型，需要是指令模型，可以和TTT的基座模型不同。
- `pad_to_max_length`：是否把TTT的所有输入数据pad到 `model_max_length`，通常不需要。
- `ttt_recite_first`：TTT时是否要求模型先输出evidence，再输出答案。这个参数会影响synthetic QA的prompt和输出。
- `ttt_enable_ICL`：TTT时是否启用ICL。这个参数会影响synthetic QA的输入。
- `qa_loss_weight`：TTT时synthetic QA的损失权重，反映为单条synthetic QA的损失权重是单条长文本片段的损失权重的 `loss_qa_weight` 倍。取较大值可以强调synthetic QA的重要性。
- `enable_diverse_qa`：是否使TTT的不同epoch中的synthetic QA不同。从直观上看，启用可以防止过拟合。
- `num_timeline_reorder`：生成时间排序类synthetic QA的数量，当大于0时需要指定 `generator_name_or_path`。
- `num_timeline_reorder_events`：如果生成时间排序类synthetic QA，则该参数指定每条QA中需要排序的事件数量。该参数可以是单个整数，表示所有QA的事件数量都是该值，也可以是两个整数表示一个区间，事件数量将在区间中均匀采样。
- `append_question`：TTT时，是否在长文本片段的末尾添加测试时的问题描述。若启用，则对于测试时的不同问题，需要单独TTT，目前只支持时间排序类问题。

### `CustomTrainingArguments`
TTT所用训练参数，与 `transformers.TrainingArguments` 独立。

- `use_lora`：是否启用LoRA。
- `lora_rank`
- `load_in_4bit`
- `load_in_8bit`
- `full_ft`：是否全参数微调。
- `gather_batches`：是否将所有训练样例放入一个batch（整个epoch只更新一次）。
- `involve_qa_epochs`：在长文本上训练后，引入多少epoch长文本和synthetic QA的混合训练。和 `transformers.TrainingArguments.training_epochs` 混合使用，`training_args` 指定仅在长文本上训练的epoch数，`involve_qa_epochs` 指定在长文本和synthetic QA上混合训练的epoch数。

### `GlobalTestArguments`
通用的测试参数，通常不同的数据集有不同的测试参数，可以继承自 `GlobalTestArguments`。

- `input_file`：输入文件的位置。
- `eval_batch_size`
- `compute_attention`：是否计算attention（目前只支持LooGLE）。
- `attention_output_dir`：`compute_attention=True` 时，指定输出attention图像的目录；不指定时则不输出图像。
- `overwrite`：当检测到输出文件已经存在时，是否重写输出文件；若不重写，则会检测输出文件中已经回答的文章数量，从第一个未回答的文章开始继续测试。
- `enable_ICL`：测试时是否启用ICL。
- `recite_first`：测试时是否要求模型先输出evidence再输出答案。

### `TrainingArguments`
transformers库所需参数。

- `output_dir`：在本代码中无意义。
- `per_device_train_batch_size`
- `learning_rate`
- `weight_decay`
- `adam_beta1`, `adam_beta2`, `adam_epsilon`
- `max_grad_norm`
- `num_train_epochs`：不引入syn. QA的训练epoch数


## 实验结果文件  
实验配置在long-context文档里和overleaf上都有同步。
所有的实验结果原版输出文件：
- loogle longqa llama3+ttt: output/ttt-input8000-longqa-gpt.json
- loogle shortqa llama3+ttt: output/ttt-input8000-shortqa-gpt.json
- loogle longqa llama3+ttt without ICL: output/ttt-longqa-gpt.json
- loogle longqa llama3: output/Llama3-8B-input8000-longqa-gpt.json
- loogle shortqa llama3: output/Llama3-8B-input8000-shortqa-gpt.json
- gpt3.5+ttt longqa: output/gpt3.5_assis_only_b2048_o1024_lrm0.5_epoch2_longqa.json
- gpt3.5 longqa: output/gpt3.5_baseline_134.json
- gpt3.5+ttt shortqa: output/gpt3.5_assis_only_b2048_o1024_lrm0.5_epoch3_shortqa.json
- gpt3.5 shortqa: 直接采用loogle结果，因为实验setting等都未发生变化
- loogle longqa llama3+ttt 直接给evidence: output/ttt_evidence_longqa.json
- loogle longqa llama3 直接给evidence: output/llama3_with_evidence.json

剩余文件里名称近似的多为计算了分数的输出文件，后面会有介绍。也可以自行查看文件内容。
需要注意的是，上述部分文件格式可能和接下来列出的现在代码文件的输出文件格式不符合，因为在实验过程中代码发生了改动。

## 获取llama3系列的output

使用shells/examples/loogle.sh获取模型的输出，注意以下参数：
- num_train_epochs: 等于0时相当于测试没有经过ttt的原模型
- full_ft: 是否进行全参数微调（之前的版本如果为false的话默认使用lora）
- block_size: 现在的配置为1024
- len_segment: 现在配置为2
- len_offset: 现在配置为1
- output_file: 每次必须更改，最好名称反映参数配置。目前所有输出文件都在output文件夹下。
- enable_ICL: 是否采用ICL
- recite_first: 是否先prompt模型回答evidence
- dataset_name: long_qa或者short_qa
- debug_size: 测试的数据条数
- compute_attention: 是否计算attention  

任务运行后的output文件为json格式，相较于输入文件在每对qa pair里添加了pred部分存储模型输出，内部格式为：
`[
    {
        "title":...
        "output":"none",
        "qa_pairs": [
            {
                "S":...
                "Q":...
                "A":...
                "pred":...
            },
            ...
        ]
    },
    ...
]`


使用shells/examples/bamboo.sh获取模型输出，需要注意的参数基本和loogle一样，需要额外注意的是：
- input_file: 目前只支持datasets/bamboo/reportsumsort_16k.jsonl
- prompt_name: 目前只支持reportsumsort
- prompt_path: 默认为scripts/prompt_bamboo.json不需要更改  
  
任务运行后的输出文件为jsonl格式，每一行数据格式如下：
`
{"pred": [...], "answer":[...], "output":模型原始输出}
`

## 获取gpt3.5的output

可复用的训练gpt3.5的代码文件为gpt.py，里面有两个类，FTDataset处理数据，FTGPT进行微调；两个函数，check_format检查训练文件格式是否符合微调gpt的要求，dump_data用于把ftdataset生成的训练数据放进一个文件里形成训练文件（微调gpt必须上传一个特定格式的文件）。具体类的方法和函数的使用见gpt.py内的注释。为了方便观察gpt微调过程中的各种情况，这个文件内部函数都添加了logging的部分，所以请在使用时提前设置好logging，例如接下来提到的loogle.py和bamboo.py在文件开头都设置了logging的config。


对于loogle，使用scripts/gpt3.5/loogle.py文件，需要注意的参数如下：
- dataset: 测试用文件位置，例如datasets/loogle/shortdep_qa.jsonl
- debug_size: 选取数据集样例数
- block_size: 和llama3部分的block_size不同，这里相当于论文初稿里定义的L
- offset: 和llama3部分的offset不同，这里相当于论文初稿里定义的S
- n_epochs: epoch数
- do_eval: 是否prompt模型生成输出文件
- output_file: 输出文件名
- resume: 从该文件里加载存储的先前微调过的模型测试输出。
- resume_file: pkl文件，如果resume为true，那么读取改文件内容使用。需要注意的是，这个文件同时也会存储新训练的模型，所以如果想保留文件里原本存储之前训练过的模型，一定要传递resume参数！！！
- baseline: 想要直接测试原始的gpt3.5则传递该参数，记得同时传递output_file

任务运行后的输出文件为json格式，与llama3的loogle部分格式相同。

对于bamboo，运行scripts/gpt3.5/bamboo.py文件，参数和上述loogle.py基本一致。
任务运行后输出文件格式和llama3的bamboo部分相同。

ps: 目前已经存在的pkl文件：long.pkl存储了在longqa上测试的134篇文章分别对应的模型，该134篇文章对应的文件为datasets/loogle/gpt3.5_test.jsonl；short.pkl存储了shortqa上所有文章对应的训练后的模型；bamboo.pkl里存储了在bamboo的reportsumsort上训练的63篇文章对应的模型。

## 计算分数  
### meteor & bertscore  
运行scripts/test/get_auto_scores.py，参数：
- input_file: 必须满足前文介绍的loogle部分输出文件格式
- output_file: 输出json文件里每条数据格式如下：
`{
    "title":...
    "output":"none",
    "qa_pairs": [
        {
            "S":...
            "Q":...
            "A":...
            "pred":...
            "scores":{
                "meteor":...,
                "bertscore":...,
                ...
            }
        },
        ...
    ]
}
`  

需要注意的是，如果输出文件每条数据里本身存在scores键值则append否则创建。
另外，会根据input_file的名字输出一个加上auto_score后缀的json文件，包含在整个数据集上的meteor和bertscore的平均值。

### gpt4 score
运行scripts/test/get_gpt4_score.py，用法和get_auto_scores.py基本相同。
不同之处是，最终会根据input_file的名字输出一个加上gpt4_score后缀的txt文件，包含在整个数据集上的gpt4 score的平均值。

### coordinate index
运行scripts/test/get_CI_score.py，参数：
- input: 必须满足前文介绍的bamboo部分输出文件格式
直接在terminal给出CI分数

## 剩余一些可借鉴的部分
在scripts/datasets里有一些之前用来对数据集进行格式预处理、后处理，生成qa，计算gpt训练所需总token数目，计算不同type的longqa分别的准确率的代码。虽然因为比较简单且不常用没做模块化，可复用差，但如果需要可以借鉴改一下之后使用。