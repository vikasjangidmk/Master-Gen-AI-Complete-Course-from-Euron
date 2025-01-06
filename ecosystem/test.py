from langserve import RemoteRunnable

chain = RemoteRunnable("http://localhost:8000/chain/c/N4XyA/playground")

res = chain.invoke({"language":"hindi","text":"What is generative ai ?"})

print(res)