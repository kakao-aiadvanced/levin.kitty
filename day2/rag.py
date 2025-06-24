from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

def load_documents(urls: list[str]) -> list[Document]:
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return docs

def split_documents(docs: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(docs)
    return texts

def create_vector_store(texts: list[Document]) -> BaseRetriever:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )
    vector_store = Chroma(
        collection_name="lilianweng",
        embedding_function=embeddings,
    )
    vector_store.add_documents(texts)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )
    return retriever

# 5. 관련성 평가

prompt_template = PromptTemplate(
    input_variables=["query", "document_chunk"],
    template="""
사용자 쿼리와 검색된 문서 청크 간의 관련성을 평가해주세요.

**사용자 쿼리:** {query}

**검색된 문서 청크:** {document_chunk}

**지시사항:**
- 문서 청크가 사용자 쿼리와 직접적으로 관련이 있거나 쿼리에 답변할 수 있는 정보를 포함하고 있다면 관련있음으로 판단
- 문서 청크가 사용자 쿼리와 전혀 다른 주제이거나 답변에 도움이 되지 않는다면 관련없음으로 판단

**출력 형식:**
반드시 다음 JSON 형식으로만 응답하세요:
{{"relevance": "yes"}} 또는 {{"relevance": "no"}}

다른 설명이나 추가 텍스트 없이 위 JSON 형식으로만 응답해주세요.
"""
)

def filter_relevant_documents(query, documents, chain):
    """
    쿼리와 관련성이 있는 문서들만 필터링해서 반환하는 함수
    
    Args:
        query (str): 사용자 쿼리
        documents (list): 검색된 문서 리스트
        chain: 관련성 평가 체인
    
    Returns:
        list: 관련성이 있는 문서들의 리스트
    """
    relevant_docs = []
    
    for i, doc in enumerate(documents, 1):
        result = chain.invoke({"query": query, "document_chunk": doc.page_content})
        print(f"\n[문서 {i}] 관련성: {result}")
        
        if result.get('relevance') == 'yes':
            relevant_docs.append(doc)
            print(f"✅ 관련 문서로 선택됨")
        else:
            print(f"❌ 관련성 없음")
        print("-" * 30)
    
    return relevant_docs

# 6. 케이스 
# All Yes Case: query = "LLM"
# All Non Case: query = "Jeju Island"    

# 8. RAG 답변 생성
rag_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
다음 문서들을 참고하여 질문에 답변해주세요.

**참고 문서:**
{context}

**질문:** {question}

**지시사항:**
- 위의 참고 문서들만을 기반으로 답변해주세요
- 문서에 없는 내용은 추측하지 말고 "문서에서 해당 정보를 찾을 수 없습니다"라고 해주세요
- 한국어로 자세하고 명확하게 답변해주세요
- 가능하면 참고한 문서의 내용을 인용해주세요
- 답변 마지막에 **출처** 섹션을 만들어 참고한 문서들의 URL을 나열해주세요

**답변:**
"""
)

def generate_rag_answer(query, relevant_docs, llm):
    """
    관련 문서들을 사용해서 RAG 답변을 생성하는 함수
    
    Args:
        query (str): 사용자 질문
        relevant_docs (list): 관련성 있는 문서들
        llm: 언어 모델
    
    Returns:
        str: 생성된 답변
    """
    if not relevant_docs:
        return "관련성 있는 문서를 찾을 수 없어서 답변을 생성할 수 없습니다."
    
    # 관련 문서들을 컨텍스트로 합치기 (출처 URL 포함)
    context_parts = []
    source_urls = set()  # 중복 제거를 위한 set
    
    for i, doc in enumerate(relevant_docs):
        source_url = doc.metadata.get('source', 'Unknown')
        source_urls.add(source_url)
        context_parts.append(f"[문서 {i+1}]\n{doc.page_content}\n출처: {source_url}")
    
    context = "\n\n".join(context_parts)
    
    # 출처 URL 목록 추가
    if source_urls and 'Unknown' not in source_urls:
        context += f"\n\n**참고 문서 URL 목록:**\n"
        for url in sorted(source_urls):
            context += f"- {url}\n"
    
    rag_chain = rag_template | llm
    response = rag_chain.invoke({"context": context, "question": query})
    
    return response.content if hasattr(response, 'content') else str(response)

# 9. Hallucination 체크 및 재생성
hallucination_template = PromptTemplate(
    input_variables=["context", "question", "answer"],
    template="""
제공된 문서들과 생성된 답변을 비교하여, 답변에 Hallucination(환각)이 있는지 확인해주세요.

**원본 문서들:**
{context}

**질문:** {question}

**생성된 답변:** {answer}

**평가 기준:**
1. 답변의 모든 내용이 제공된 문서들에서 직접 확인 가능한가?
2. 답변에 문서에 없는 추가 정보나 추측이 포함되어 있는가?
3. 답변이 문서의 내용을 정확히 반영하고 있는가?

**출력 형식:**
반드시 다음 JSON 형식으로만 응답하세요:
{{"hallucination": "yes/no", "explanation": "한국어로 판단 이유 설명", "confidence": "high/medium/low"}}

다른 설명이나 추가 텍스트 없이 위 JSON 형식으로만 응답해주세요.
"""
)

def check_hallucination(context, question, answer, llm):
    """
    RAG 답변에 Hallucination이 있는지 확인하는 함수
    
    Args:
        context (str): 원본 문서 컨텍스트
        question (str): 사용자 질문
        answer (str): 생성된 답변
        llm: 언어 모델
    
    Returns:
        dict: Hallucination 평가 결과
    """
    hallucination_chain = hallucination_template | llm | JsonOutputParser()
    result = hallucination_chain.invoke({
        "context": context, 
        "question": question, 
        "answer": answer
    })
    return result

def generate_rag_with_hallucination_check(query, relevant_docs, llm, max_retries=1):
    """
    Hallucination 체크와 함께 RAG 답변을 생성하는 함수 (재시도 포함)
    
    Args:
        query (str): 사용자 질문
        relevant_docs (list): 관련성 있는 문서들
        llm: 언어 모델
        max_retries (int): 최대 재시도 횟수
    
    Returns:
        tuple: (최종_답변, hallucination_결과들)
    """
    if not relevant_docs:
        return "관련성 있는 문서를 찾을 수 없어서 답변을 생성할 수 없습니다.", []
    
    context = "\n\n".join([
        f"[문서 {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(relevant_docs)
    ])
    
    hallucination_results = []
    current_answer = generate_rag_answer(query, relevant_docs, llm)
    
    for attempt in range(max_retries + 1):
        print(f"\n🔍 Hallucination 체크 (시도 {attempt + 1}):")
        print("=" * 60)
        
        # Hallucination 체크
        hallucination_result = check_hallucination(context, query, current_answer, llm)
        hallucination_results.append({
            'attempt': attempt + 1,
            'result': hallucination_result
        })
        
        print(f"Hallucination 여부: {hallucination_result['hallucination']}")
        print(f"신뢰도: {hallucination_result['confidence']}")
        print(f"설명: {hallucination_result['explanation']}")
        
        if hallucination_result['hallucination'] == 'no':
            print("✅ 답변이 문서 기반으로 정확합니다.")
            break
        else:
            print("⚠️  답변에 Hallucination이 감지되었습니다!")
            
            # 재시도 가능한 경우
            if attempt < max_retries:
                print(f"🔄 답변을 재생성합니다... (재시도 {attempt + 1}/{max_retries})")
                
                # 재생성용 개선된 프롬프트 사용
                improved_rag_template = PromptTemplate(
                    input_variables=["context", "question", "previous_issues"],
                    template="""
다음 문서들을 참고하여 질문에 답변해주세요.

**참고 문서:**
{context}

**질문:** {question}

**이전 답변의 문제점:** {previous_issues}

**중요한 지시사항:**
- 반드시 제공된 문서들에서만 정보를 가져와주세요
- 문서에 명시되지 않은 내용은 절대 추가하지 마세요
- 추측이나 일반 상식을 바탕으로 한 정보는 포함하지 마세요
- 문서에서 직접 확인할 수 있는 내용만 사용해주세요
- 답변 마지막에 **출처** 섹션을 만들어 참고한 문서들의 URL을 나열해주세요

**답변:**
"""
                )
                
                improved_chain = improved_rag_template | llm
                current_answer = improved_chain.invoke({
                    "context": context, 
                    "question": query,
                    "previous_issues": hallucination_result['explanation']
                }).content if hasattr(improved_chain.invoke({
                    "context": context, 
                    "question": query,
                    "previous_issues": hallucination_result['explanation']
                }), 'content') else str(improved_chain.invoke({
                    "context": context, 
                    "question": query,
                    "previous_issues": hallucination_result['explanation']
                }))
                
                print(f"\n🤖 재생성된 답변:")
                print("=" * 60)
                print(current_answer)
                print("=" * 60)
            else:
                print(f"❌ 최대 재시도 횟수에 도달했습니다. 마지막 답변을 사용합니다.")
        
        print("=" * 60)
    
    return current_answer, hallucination_results

if __name__ == "__main__":
    
    # 문서 링크
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    
    # 쿼리
    query = "Jeju Island"
    query = "agent memory"
    
    print(f"🔍 쿼리: '{query}'")
    print("=" * 60)
    
    # 1. 문서 로드 및 분할
    print("📁 문서 로드 중...")
    docs = load_documents(urls)
    texts = split_documents(docs)
    print(f"✅ {len(texts)}개 문서 청크 생성 완료")
    
    # 2. 벡터 스토어 생성 및 검색
    print("🔍 벡터 스토어 생성 및 문서 검색 중...")
    retriever = create_vector_store(texts)
    retrieved_documents = retriever.invoke(query)
    print(f"✅ {len(retrieved_documents)}개 문서 검색 완료")
    
    # 3. 관련성 평가 및 필터링
    print("\n📋 관련성 평가 시작...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    chain = prompt_template | llm | JsonOutputParser()
    relevant_documents = filter_relevant_documents(query, retrieved_documents, chain)
    
    print(f"\n📊 결과: 전체 {len(retrieved_documents)}개 문서 중 {len(relevant_documents)}개가 관련성 있음")
    
    # 4. RAG 답변 생성 (Hallucination 체크 및 재생성 포함)
    if relevant_documents:
        print(f"\n🤖 '{query}'에 대한 RAG 답변 생성 시작...")
        print("=" * 60)
        
        final_answer, hallucination_history = generate_rag_with_hallucination_check(
            query, relevant_documents, llm, max_retries=1
        )
        
        print(f"\n📝 최종 답변:")
        print("=" * 60)
        print(final_answer)
        print("=" * 60)
        
        # 5. 최종 결과 요약
        print(f"\n📊 최종 결과 요약:")
        print("=" * 60)
        print(f"총 시도 횟수: {len(hallucination_history)}")
        for i, result in enumerate(hallucination_history):
            status = "✅ 통과" if result['result']['hallucination'] == 'no' else "⚠️  Hallucination 감지"
            print(f"시도 {result['attempt']}: {status}")
        
        if hallucination_history[-1]['result']['hallucination'] == 'no':
            print("🎉 최종 답변이 검증되었습니다!")
        else:
            print("⚠️  최종 답변에 여전히 Hallucination이 있을 수 있습니다.")
        print("=" * 60)
    else:
        print("\n⚠️  관련성 있는 문서가 없어 RAG 답변을 생성할 수 없습니다.")
     