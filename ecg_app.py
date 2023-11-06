from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain import PromptTemplate
import torch
import streamlit as st

@st.cache_resource
def retrieve():
    from trafilatura import fetch_url, extract

    url = "https://ja.m.wikipedia.org/wiki/%E5%BF%83%E9%9B%BB%E5%9B%B3"
    filename = 'textfile.txt'

    document = fetch_url(url)
    text = extract(document)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)

    loader = TextLoader(filename, encoding='utf-8')
    documents = loader.load()

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator = "\n",
        chunk_size=800,
        chunk_overlap=20,
    )
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    db = FAISS.from_documents(texts, embeddings)

    # 一番類似するチャンクをいくつロードするかを変数kに設定出来ます。
    retriever = db.as_retriever(search_kwargs={"k": 3})

    return retriever

retriever = retrieve()

@st.cache_resource
def tokenizer_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    model_id = "elyza/ELYZA-japanese-Llama-2-7b-instruct"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config
        ).eval()
        #,
        #offload_folder="path/to/offload_folder"  # offload_folderを指定
    return tokenizer, model

tokenizer, model = tokenizer_model()

@st.cache_resource
def template():
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人の循環器内科医です。step by step に考え、正確に回答してください。"
    text = "{context}\n質問: {question}"
    template = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
        bos_token=tokenizer.bos_token,
        b_inst=B_INST,
        system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
        prompt=text,
        e_inst=E_INST,
    )
    return template

template = template()


@st.cache_resource
def pipe_qa():
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
    )
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
        template_format="f-string"
    )


    chain_type_kwargs = {"prompt":prompt}

    qa = RetrievalQA.from_chain_type(
        llm=HuggingFacePipeline(
            pipeline=pipe,
            model_kwargs=dict(
                temperature=0.0,
                do_sample=True,
                max_length=1024,
                repetition_penalty=2
            )
        ),
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True,
    )
    return qa

qa = pipe_qa()

image_path = '/content/スクリーンショット 2023-11-06 120138.png'
st.title('心電図特化LLM')
st.image(image_path, use_column_width=True)
st.write('循環器内科医になりきってお答えします。')
st.write('多忙ゆえ、回答に時間がかかる場合がございます。ご了承ください。')
question = st.text_input("質問を入力してください", "")

# "生成" ボタン
if st.button("回答を見る"):
    # 質問に回答を生成
    result = qa(question)
    st.write(result['result'])

    # 不要なGPUメモリを解放
    torch.cuda.empty_cache()

