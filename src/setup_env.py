import os

def setup():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    llm_models_dir = os.path.join(base_dir, "llm_models")
    
    subfolders = ["qwen", "mistral", "llama"]
    
    print(f"Scaffolding model directories in {llm_models_dir}...")
    
    for folder in subfolders:
        path = os.path.join(llm_models_dir, folder)
        os.makedirs(path, exist_ok=True)
        # Create .gitkeep to ensure directory is tracked
        with open(os.path.join(path, ".gitkeep"), "w") as f:
            pass
        print(f"  - Created {folder}/.gitkeep")
        
    print("\nSetup complete. Place your .gguf files in the corresponding subfolders.")

if __name__ == "__main__":
    setup()
