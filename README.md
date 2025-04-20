<pre>
my_llm_tool/
├── my_llm_tool/
    ├── api.py
    ├── task_manager.py
    ├── model_loader.py
    ├── batch_manager.py
</pre>

<pre>'''
my_llm_tool/
├── my_llm_tool/
    ├── api.py
    ├── task_manager.py
    ├── model_loader.py
    ├── batch_manager.py
'''</pre>

## 🔧 Installing AutoAWQ (for quantized model inference)

`autoawq` **must be installed manually** to avoid dependency resolution errors.

### Step 1: Ensure torch is already installed (from requirements.txt)

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
