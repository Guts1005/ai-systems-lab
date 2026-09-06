from vllm.config.speculative import SpeculativeConfig

for field_name, field in SpeculativeConfig.__pydantic_fields__.items():
    print(f"{field_name}: {field.annotation} (default: {field.default})")
