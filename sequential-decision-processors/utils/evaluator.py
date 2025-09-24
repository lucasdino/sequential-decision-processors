import os, time, json, asyncio, wandb
from collections     import defaultdict
from typing          import Any

from .results_dict   import ResultsDict, DifficultyResultsDict
from .dataclass      import JSONLDataClass


class Evaluator():
    def __init__(self, args, task_map):
        """ Given a set of eval_files instantiate an evaluator object to analyze the evals. """
        self.args = args
        self.task_map = task_map
        self._setup_wandb()
        
        # Load in our various data files
        self.dataclasses = [JSONLDataClass(args.data_dir, filename, task_map, args.model_version) for filename in args.data_files]

        # Setup various vals just once
        os.makedirs(os.path.join(args.data_dir, 'saved_data'), exist_ok=True)
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
                    
    def evaluate(self, model):
        """ 
        Evaluate the model on the eval files. 
        model: The vLLM model client to evaluate.
        """
        result_dicts = []

        # Loop through all our dataclasses and generate / evaluate
        for dataclass in self.dataclasses:
            verbose_generations = []
            
            # Initial setup
            data = dataclass.data
            max_len = len(data) if self.args.max_samples is None else min(len(data), self.args.max_samples)
            print(f"{'='*50}\n Evaluating: {dataclass.trimmed_filename} for {max_len} samples:\n{'='*50}")

            # Set up results dict
            results = ResultsDict(
                task_type = dataclass.task_type,
                filename = dataclass.filename,
                wandb_run = self.wandb_run
            )
            
            # Main eval loop per dataclass
            for start_idx in range(0, max_len, self.args.batch_size):
                data_batch = data[start_idx:min(start_idx+self.args.batch_size, max_len)]
                prompts = [datum['prompt'] for datum in data_batch]
                batch_responses = asyncio.run(model.chat(prompts))

                # Now add results / append response to your dataset
                for idx in range(len(data_batch)):
                    prompt = data_batch[idx]['prompt']
                    response = batch_responses[idx]
                    prompt_info = data_batch[idx]['info']
                    ground_truth = data_batch[idx]['info']['answer']

                    results.add_result(prompt, response, prompt_info)

                    # Optionally log responses to console for visibility                    
                    if self.args.verbose:
                        print(f"{'-'*10}\nPrompt:\n{prompt}\n")
                        print(f"Model Response:\n{response}\nGround Truth Answer:\n'{ground_truth}'\n")
                    if self.args.save_verbose:
                        verbose_generations.append({
                            "prompt": prompt,
                            "model_response": response,
                            "info": prompt_info
                        })

            results, correct_responses = results.get_final_dict(self.args.run_type)
            result_dicts.append(results)
            
            # Finally print results from dataclass evaluation
            print(f"{'-'*50}\nResults for {dataclass.filename}:")
            for key, value in result_dicts[-1].items():
                print(f"{key}: {value}")
            print(f"{'-'*50}\n\n")

            # Save our correct responses if 'rejsampling' task
            if self.args.run_type == 'rejsampling':
                save_path = os.path.join(dataclass.data_dir, 'saved_data', f"{dataclass.trimmed_filename}_correct_{self.timestamp}.json")
                with open(save_path, 'w') as f:
                    json.dump(correct_responses, f, indent=4)
            
            # Also save if save_verbose
            if self.args.save_verbose:
                save_path = os.path.join(dataclass.data_dir, 'saved_data', f"{dataclass.trimmed_filename}_all_{self.timestamp}.json")
                with open(save_path, 'w') as f:
                    json.dump(verbose_generations, f, indent=4)

        return result_dicts

    # ============================
    # Setup Helpers
    # ============================

    def _setup_wandb(self):
        # Set up wandb logger
        if self.args.use_wandb:
            self.wandb_run = wandb.init(
                name = self.args.experiment_name,
                config={
                    "model": self.args.model,
                    "temperature": self.args.temperature,
                    "top_p": self.args.top_p,
                    "min_p": self.args.min_p,
                    "top_k": self.args.top_k,
                    "repetition_penalty": self.args.repetition_penalty,
                }
            )
        else:
            self.wandb_run = None



# =============================================
# N_Evaluator: 
# Evaluator class that generates 'args.num_generations' 
# responses and uses concurrency.
# =============================================
class N_Evaluator:
    """
    Run up‑to‑N generations per sample, fully async, keeping the vLLM
    endpoint saturated.  Results are accumulated in a DifficultyResultsDict
    and `board_id_results` is dumped to disk after every file.
    """
    def __init__(self, args, task_map):
        self.args        = args
        self.task_map    = task_map
        self.num_gens    = args.num_generations          # ← how many per sample
        self.batch_size  = args.batch_size               # ← max inflight calls
        self._setup_wandb()

        self.dataclasses = [
            JSONLDataClass(args.data_dir, f, task_map, args.model_version)
            for f in args.data_files
        ]

        os.makedirs(os.path.join(args.data_dir, "saved_data"), exist_ok=True)
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")

    # ------------------------------------------------------------------ #
    def evaluate(self, model):
        return asyncio.run(self._evaluate_async(model))

    # ------------------------------------------------------------------ #
    async def _evaluate_async(self, model):
        result_dicts = []

        for dc in self.dataclasses:
            print(f"{'='*60}\n Evaluating {dc.trimmed_filename}\n{'='*60}")
            rd         = DifficultyResultsDict(dc.task_type, dc.filename, self.wandb_run)
            board_cnt  = defaultdict(int)          # board_id → #calls so far
            lock       = asyncio.Lock()            # guards rd + board_cnt
            sema       = asyncio.Semaphore(self.batch_size)

            handle_tasks = []                      # track every generation

            # ---------------------------------------------------------- #
            async def _request(prompt: str) -> str:
                return (await model.chat([prompt]))[0]

            # ---------------------------------------------------------- #
            async def _handle(prompt: str, info: dict[str, Any]):
                async with sema:
                    try:
                        resp = await _request(prompt)
                        async with lock:
                            rd.add_result(resp, info)
                            board_cnt[info["board_id"]] += 1
                    except Exception:
                        async with lock:
                            rd.results["Error: Other"] += 1

            # ---------------------------------------------------------- #
            async def _worker(datum: dict[str, Any]):
                info   = datum["info"]
                prompt = datum["prompt"]

                for _ in range(self.num_gens):
                    task = asyncio.create_task(_handle(prompt, info))
                    handle_tasks.append(task)

            # ---------------------------------------------------------- #
            # Schedule all workers, then await every generation task
            await asyncio.gather(*(_worker(d) for d in dc.data))
            await asyncio.gather(*handle_tasks)

            # ---------------- wrap‑up per file ----------------------- #
            result_dicts.append(rd.get_final_dict())

            print(f"{'-'*50}\nResults for {dc.filename}:")
            for k, v in result_dicts[-1].items():
                print(f"{k}: {v}")
            print(f"{'-'*50}\n")

            save_path = os.path.join(
                dc.data_dir,
                "saved_data",
                f"{dc.trimmed_filename}_board-id-results_{self.timestamp}.json",
            )
            with open(save_path, "w") as f:
                json.dump(rd.board_id_results, f, indent=2)

            print(f"{'-'*48}\nResults ({dc.trimmed_filename}) saved → {save_path}\n{'-'*48}")

        return result_dicts

    # ============================
    # Setup Helpers
    # ============================
    def _setup_wandb(self):
        if self.args.use_wandb:
            self.wandb_run = wandb.init(
                name   = self.args.experiment_name,
                config = {
                    "model"              : self.args.model,
                    "temperature"        : self.args.temperature,
                    "top_p"              : self.args.top_p,
                    "min_p"              : self.args.min_p,
                    "top_k"              : self.args.top_k,
                    "repetition_penalty" : self.args.repetition_penalty,
                },
            )
        else:
            self.wandb_run = None



# =============================================
# Simple class to allow arbitrary generation
# =============================================
class Generator():
    """
    Simplified version that just does generation aned saves down results from model inference without doing any checks.
    """
    def __init__(self, args, task_map):
        """ Given a set of eval_files instantiate an evaluator object to analyze the evals. """
        self.args = args
        self.task_map = task_map
        
        # Load in our various data files
        self.dataclasses = [JSONLDataClass(args.data_dir, filename, task_map, args.model_version) for filename in args.data_files]

        # Setup various vals just once
        os.makedirs(os.path.join(args.data_dir, 'saved_data'), exist_ok=True)
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
                    
    def generate(self, model):
        """ 
        Evaluate the model on the eval files. 
        model: The vLLM model client to evaluate.
        """
        # Loop through all our dataclasses and generate / evaluate
        for dataclass in self.dataclasses:
            verbose_generations = []
            
            # Initial setup
            data = dataclass.data
            max_len = len(data) if self.args.max_samples is None else min(len(data), self.args.max_samples)
            print(f"{'='*50}\n Evaluating: {dataclass.trimmed_filename} for {max_len} samples:\n{'='*50}")

            # Main eval loop per dataclass
            for start_idx in range(0, max_len, self.args.batch_size):
                data_batch = data[start_idx:min(start_idx+self.args.batch_size, max_len)]
                prompts = [datum['prompt'] for datum in data_batch]
                batch_responses = asyncio.run(model.chat(prompts))

                # Now add results / append response to your dataset
                for idx in range(len(data_batch)):
                    prompt = data_batch[idx]['prompt']
                    response = batch_responses[idx]
                    prompt_info = data_batch[idx]['info']

                    # Optionally log responses to console for visibility                    
                    if self.args.verbose:
                        print(f"{'-'*10}\nPrompt:\n{prompt}\n")
                        print(f"Model Response:\n{response}\n")
                    if self.args.save_verbose:
                        verbose_generations.append({
                            "prompt": prompt,
                            "model_response": response,
                            "info": prompt_info
                        })
            
            # Also save if save_verbose
            if self.args.save_verbose:
                save_path = os.path.join(dataclass.data_dir, 'saved_data', f"{dataclass.trimmed_filename}_all_{self.timestamp}.json")
                with open(save_path, 'w') as f:
                    json.dump(verbose_generations, f, indent=4)
