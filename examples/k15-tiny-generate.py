from vllm import LLM, SamplingParams

def main():
    model = "facebook/opt-125m"
    print("loading", model)

    llm = LLM(
        model=model,
        dtype="float32",
        max_model_len=512,
        gpu_memory_utilization=0.50,
        enforce_eager=True,
    )

    params = SamplingParams(max_tokens=32, temperature=0.0)
    out = llm.generate(["The capital of Texas is"], params)

    print("OUTPUT:")
    print(out[0].outputs[0].text)

if __name__ == "__main__":
    main()
