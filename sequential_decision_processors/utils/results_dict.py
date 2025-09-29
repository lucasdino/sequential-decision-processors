import re
import math

from .exceptions import ParseException, IllegalMoveException
from .parsing import extract_solution, coerce_response


class ResultsDict():
    def __init__(self, task_type, filename, wandb_run):
        self.task_type = task_type
        self.filename = filename
        self.trimmed_filename = filename.split("_", 1)[0]
        self.wandb_run = wandb_run
        self.results = self._instantiate_dict()
        self.correct_responses = []

    def add_result(self, prompt, model_response, info):
        try:
            self.results["Total Samples"] += 1
            ground_truth = info['answer']
            if self.task_type == "choose_from_n":
                answer = ground_truth['answer']
                candidates = ground_truth['candidates']
                predicted_answer = coerce_response(extract_solution(model_response), self.task_type)
                
                # Determine correctness
                if predicted_answer == answer:
                    self.results["Correct"] += 1
                    self.correct_responses.append({
                        "prompt": prompt,
                        "model_response": model_response,
                        "info": info
                    })
                else:
                    if predicted_answer in candidates:
                        self.results["Incorrect"] += 1
                    else:
                        raise IllegalMoveException("Predicted move is not in the provided moves.")
            
            elif self.task_type == 'produce_list':
                answer = ground_truth
                self.results["Total Ground Truth Legal Moves"] += len(answer)
                predicted_answer = coerce_response(extract_solution(model_response), self.task_type)

                # Compute correctness
                num_right = 0
                already_guessed = set()
                for move in predicted_answer:
                    if move in answer and move not in already_guessed:
                        already_guessed.add(move)
                        num_right += 1
                        self.results["Predicted Ground Truth Legal Moves"] += 1
                    else:
                        self.results["Illegal Moves"] += 1

                # Only append to correct_response if score >= 0.6
                score = num_right / (len(answer) + len(predicted_answer) - num_right)
                if score >= 0.6:
                    self.correct_responses.append({
                        "prompt": prompt,
                        "model_response": model_response,
                        "info": info
                    })
                
            elif self.task_type == 'predict_singlemove':
                answer = ground_truth

                predicted_answer = coerce_response(extract_solution(model_response), self.task_type)
                sorted_answers = sorted(answer.items(), key=lambda x: x[1])
                
                if predicted_answer in answer:
                    self.results["Legal Moves Provided"] += 1
                    predicted_move_idx = next(i for i, (move, _) in enumerate(sorted_answers) if move == predicted_answer)
                    self.results["Cumulative Rank of Moves Provided"] += predicted_move_idx/len(sorted_answers)

                    # Only keep if > 0.7 (w/in top 30% of moves)
                    # if predicted_move_idx/len(sorted_answers) > 0.7:
                    if predicted_move_idx/len(sorted_answers) >= 0.0:
                        self.correct_responses.append({
                            "prompt": prompt,
                            "model_response": model_response,
                            "info": info
                        })
                else:
                    raise IllegalMoveException("Predicted move is not in the legal moves.")
        
            elif self.task_type == "ntp_playmove":
                answer = ground_truth
                predicted_answer = model_response   # Since just asking it to produce a move directly (no reasoning)
                sorted_answers = sorted(answer.items(), key=lambda x: x[1])

                if predicted_answer in answer:
                    self.results["Legal Moves Provided"] += 1
                    predicted_move_idx = next(i for i, (move, _) in enumerate(sorted_answers) if move == predicted_answer)
                    self.results["Cumulative Rank of Moves Provided"] += predicted_move_idx/len(sorted_answers)
                else:
                    raise IllegalMoveException("Predicted move is not in the legal moves.")

            elif self.task_type == "ntp_yes_no":
                answer = ground_truth['answer']
                predicted_answer = model_response
                if not predicted_answer in ["Yes", "No"]:
                    raise IllegalMoveException("Output is not a 'Yes' or 'No'.")
                self.results["Total Correct"] += predicted_answer == answer

            elif self.task_type == "ntp_predict_num":
                try:
                    num_pred_ans = float(model_response)
                    num_ans = float(ground_truth['answer'])
                    delta_percent = self._safe_div((num_pred_ans - num_ans), num_ans, default_div=100)
                    self.results["Cumulative % Delta"] += delta_percent
                    self.results["Total Legal"] += 1
                    if math.isclose(num_pred_ans, num_ans):
                        self.results["Total Correct"] += 1
                except:
                    raise IllegalMoveException("Output is not able to be converted to an int.")
                
            elif self.task_type == "ntp_predict_move":
                answer = ground_truth['answer']
                predicted_answer = model_response
                if not re.fullmatch(r"[a-h][1-8]", predicted_answer):
                    raise IllegalMoveException("Output is not a correct square.")
                self.results["Total Correct"] += predicted_answer == answer

        # Exception handling to log various errors     
        except Exception as e:
            if isinstance(e, ParseException):
                self.results["Error: Parsing"] += 1
            elif isinstance(e, IllegalMoveException):
                self.results["Error: Illegal Move"] += 1
            else:
                self.results["Error: Other"] += 1
        
    def get_final_dict(self, run_type):
        """ run_type is either 'eval' or 'rejsampling' -- used for wandb logging. """
        run_type = run_type.capitalize()

        if self.task_type == "choose_from_n":
            total = self.results["Total Samples"]
            self.results["Accuracy"] = self._safe_div(self.results["Correct"], total)
            self.results["Error Rate"] = self._safe_div(self.results['Error: Parsing'] + self.results['Error: Illegal Move'] + self.results['Error: Other'], total)
            if self.wandb_run:
                self.wandb_run.log({
                    f"{run_type} - {self.trimmed_filename}/Accuracy": self.results["Accuracy"],
                    f"{run_type} - {self.trimmed_filename}/Error Rate": self.results["Error Rate"],
                })

        elif self.task_type == "produce_list":
            gt_total = self.results["Total Ground Truth Legal Moves"]
            illegal = self.results["Illegal Moves"]
            total = self.results["Total Samples"]
            self.results["Percent Legal Moves Predicted"] = self._safe_div(self.results["Predicted Ground Truth Legal Moves"], gt_total)
            self.results["Ratio of Legal to Illegal Moves"] = self._safe_div(self.results["Predicted Ground Truth Legal Moves"], illegal)
            self.results["Error Rate"] = self._safe_div(self.results['Error: Parsing'] + self.results['Error: Other'], total)
            if self.wandb_run:
                self.wandb_run.log({
                    f"{run_type} - {self.trimmed_filename}/Percent Legal Moves Predicted": self.results["Percent Legal Moves Predicted"],
                    f"{run_type} - {self.trimmed_filename}/Ratio of Legal to Illegal Moves": self.results["Ratio of Legal to Illegal Moves"],
                    f"{run_type} - {self.trimmed_filename}/Error Rate": self.results["Error Rate"]
                })

        elif self.task_type == "predict_singlemove":
            legal = self.results["Legal Moves Provided"]
            total = self.results["Total Samples"]
            self.results["Avg. Rank of Move Provided"] = self._safe_div(self.results["Cumulative Rank of Moves Provided"], legal)
            self.results["Percent Legal Moves Provided"] = self._safe_div(legal, total)
            self.results["Error Rate"] = self._safe_div(self.results['Error: Parsing'] + self.results['Error: Illegal Move'] + self.results['Error: Other'], total)
            if self.wandb_run:
                self.wandb_run.log({
                    f"{run_type} - {self.trimmed_filename}/Avg. Rank of Move Provided": self.results["Avg. Rank of Move Provided"],
                    f"{run_type} - {self.trimmed_filename}/Percent Legal Moves Provided": self.results["Percent Legal Moves Provided"],
                    f"{run_type} - {self.trimmed_filename}/Error Rate": self.results["Error Rate"]
                })

        elif self.task_type == "ntp_playmove":
            legal = self.results["Legal Moves Provided"]
            total = self.results["Total Samples"]
            self.results["Avg. Rank of Move Provided"] = self._safe_div(self.results["Cumulative Rank of Moves Provided"], legal)
            self.results["Percent Legal Moves Provided"] = self._safe_div(legal, total)
            self.results["Error Rate"] = self._safe_div(self.results['Error: Parsing'] + self.results['Error: Illegal Move'] + self.results['Error: Other'], total)
            if self.wandb_run:
                self.wandb_run.log({
                    f"{run_type} - {self.trimmed_filename}/Avg. Rank of Move Provided": self.results["Avg. Rank of Move Provided"],
                    f"{run_type} - {self.trimmed_filename}/Percent Legal Moves Provided": self.results["Percent Legal Moves Provided"],
                    f"{run_type} - {self.trimmed_filename}/Error Rate": self.results["Error Rate"]
                })

        elif self.task_type == "ntp_yes_no":
            correct_acc = self._safe_div(self.results["Total Correct"], self.results["Total Samples"])
            total_errors = self.results['Error: Illegal Move'] + self.results['Error: Other'] 
            error_rate = self._safe_div(total_errors, self.results['Total Samples'])
            self.results['% Correct'] = correct_acc
            self.results['Error Rate'] = error_rate
            if self.wandb_run:
                self.wandb_run.log({
                    f"{run_type} NTP - {self.trimmed_filename} / % Correct": self.results['% Correct'],
                    f"{run_type} NTP - {self.trimmed_filename} / Error Rate": self.results['Error Rate'],
                    })

        elif self.task_type == "ntp_predict_num":
            acc_perfect = self._safe_div(self.results["Total Correct"], self.results["Total Samples"])
            avg_delta_percent = self._safe_div(self.results["Cumulative % Delta"], self.results["Total Legal"])
            total_errors = self.results['Error: Illegal Move'] + self.results['Error: Other'] 
            error_rate = self._safe_div(total_errors, self.results['Total Samples'])
            self.results['% Perfect'] = acc_perfect
            self.results['Avg. % Delta'] = avg_delta_percent
            self.results['Error Rate'] = error_rate
            if self.wandb_run:
                self.wandb_run.log({
                    f"{run_type} NTP - {self.trimmed_filename} / Avg. % Delta": self.results['Avg. % Delta'],
                    f"{run_type} NTP - {self.trimmed_filename} / % Perfect": self.results['% Perfect'],
                    f"{run_type} NTP - {self.trimmed_filename} / Error Rate": self.results['Error Rate'],
                    })

        elif self.task_type == "ntp_predict_move":
            correct_acc = self._safe_div(self.results["Total Correct"], self.results["Total Samples"])
            total_errors = self.results['Error: Illegal Move'] + self.results['Error: Other'] 
            error_rate = self._safe_div(total_errors, self.results['Total Samples'])
            self.results['% Correct'] = correct_acc
            self.results['Error Rate'] = error_rate
            if self.wandb_run:
                self.wandb_run.log({
                    f"{run_type} NTP - {self.trimmed_filename} / % Correct": self.results['% Correct'],
                    f"{run_type} NTP - {self.trimmed_filename} / Error Rate": self.results['Error Rate'],
                    })

        return self.results, self.correct_responses

    # =================================================
    # Internal helper functions
    # =================================================
    def _instantiate_dict(self):
        if self.task_type == "choose_from_n":
            return {
                "Filename": self.filename,
                "Total Samples": 0,
                "Correct": 0,
                "Incorrect": 0,
                "Error: Parsing": 0,
                "Error: Illegal Move": 0,
                "Error: Other": 0,
            }
        elif self.task_type == "produce_list":
            return {
                "Filename": self.filename,
                "Total Samples": 0,
                "Total Ground Truth Legal Moves": 0,
                "Predicted Ground Truth Legal Moves": 0,
                "Illegal Moves": 0,
                "Error: Parsing": 0,
                "Error: Other": 0,
            }
        elif self.task_type == "predict_singlemove":
            return {
                "Filename": self.filename,
                "Total Samples": 0,
                "Legal Moves Provided": 0,
                "Cumulative Rank of Moves Provided": 0,
                "Error: Parsing": 0,
                "Error: Illegal Move": 0,
                "Error: Other": 0,
            }
        elif self.task_type == "ntp_playmove":
            return {
                "Filename": self.filename,
                "Total Samples": 0,
                "Legal Moves Provided": 0,
                "Cumulative Rank of Moves Provided": 0,
                "Error: Parsing": 0,
                "Error: Illegal Move": 0,
                "Error: Other": 0,
            }
        elif self.task_type == "ntp_yes_no":
            return {
                "Filename": self.filename,
                "Total Samples": 0,
                "Total Correct": 0,
                "Error: Illegal Move": 0,
                "Error: Other": 0
            }
        elif self.task_type == "ntp_predict_num":
            return {
                "Filename": self.filename,
                "Total Samples": 0,
                "Total Correct": 0,
                "Total Legal": 0,
                "Cumulative % Delta": 0,
                "Error: Illegal Move": 0,
                "Error: Other": 0
            }
        elif self.task_type == "ntp_predict_move":
            return {
                "Filename": self.filename,
                "Total Samples": 0,
                "Total Correct": 0,
                "Error: Illegal Move": 0,
                "Error: Other": 0
            }
        else:
            raise ValueError(f"Undefined task type: {self.task_type}")

    def _safe_div(self, x, y, default=0, default_div=None):
        if default_div is not None:
            # Use default_div if y is "close to zero"
            return x / y if not math.isclose(y, 0) else x / default_div
        else:
            return x / y if not math.isclose(y, 0) else default
    


# =============================================
# Results Dict for LLM Parsing Cases
# =============================================
class ParserResultsDict():
    def __init__(self, task_type, filename, wandb_run):
        self.task_type = task_type
        self.filename = filename
        self.trimmed_filename = filename.split("_", 1)[0]
        self.wandb_run = wandb_run
        self.results = self._instantiate_dict()

    def add_result(self, parsed_response):
        self.results["Total Responses Parsed"] += 1
        if self.task_type == "hallucination":
            for k, v in parsed_response.items():
                self.results[k] += v

        elif self.task_type == "reasoning_strategy":
            for k, v in parsed_response.items():
                self.results[f"Count: {k}"] += v
        
        elif self.task_type == "reasoning_quality":
            for k, v in parsed_response.items():
                self.results[f"Sum: Reasoning {k}"] += v

    def get_final_dict(self):
        """ Return finalized dict and log to wandb. """
        if self.task_type == "hallucination":
            total_moves = self.results['Count: Moves Checked'] + self.results['Count: Pieces Checked']
            average_moves_per_response = self._safe_div(total_moves, self.results['Total Responses Parsed'])
            moves_accuracy = self._safe_div(self.results['Count: Moves Correct'], self.results['Count: Moves Checked'])
            pieces_accuracy = self._safe_div(self.results['Count: Pieces Correct'], self.results['Count: Pieces Checked'])
            total_accuracy = self._safe_div(self.results['Count: Moves Correct'] + self.results['Count: Pieces Correct'], total_moves)
            hallucination_percent = self._safe_div(self.results['Count: Hallucinations'],  total_moves)
            percent_reprompts = self._safe_div(self.results['Error: Reprompt'], self.results['Total Responses Parsed'])

            self.results['Moves Accuracy'] = moves_accuracy
            self.results['Pieces Accuracy'] = pieces_accuracy
            self.results['Total Accuracy'] = total_accuracy
            self.results['Hallucination Percent'] = hallucination_percent
            self.results['Ave. Moves Parsed Per Response'] = average_moves_per_response
            self.results['Percent Reprompts'] = percent_reprompts
            
            if self.wandb_run:
                self.wandb_run.log({
                    f"Hallucination / Moves Accuracy": self.results['Moves Accuracy'],
                    f"Hallucination / Pieces Accuracy": self.results['Pieces Accuracy'],
                    f"Hallucination / Total Accuracy": self.results['Total Accuracy'],
                    f"Hallucination / Hallucination Percent": self.results["Hallucination Percent"],
                    f"Hallucination / Ave. Moves Parsed Per Response": self.results["Ave. Moves Parsed Per Response"],
                    f"Hallucination / Percent Reprompts": self.results["Percent Reprompts"]                
                })
        
        elif self.task_type == "reasoning_strategy":
            self.results['Percent Enumeration'] = self._safe_div(self.results['Count: Enumeration'], self.results['Total Responses Parsed'])
            self.results['Percent Tree Search'] = self._safe_div(self.results['Count: Tree Search'], self.results['Total Responses Parsed'])
            self.results['Percent Backtracking'] = self._safe_div(self.results['Count: Backtracking'], self.results['Total Responses Parsed'])
            self.results['Percent Self Correction'] = self._safe_div(self.results['Count: Self Correction'], self.results['Total Responses Parsed'])
            self.results['Percent Subgoal Setting'] = self._safe_div(self.results['Count: Subgoal Setting'], self.results['Total Responses Parsed'])
            self.results['Percent Verification'] = self._safe_div(self.results['Count: Verification'], self.results['Total Responses Parsed'])
            self.results['Percent Reprompts'] = self._safe_div(self.results['Error: Reprompt'], self.results['Total Responses Parsed'])
            
            if self.wandb_run:
                self.wandb_run.log({
                    f"Reasoning Strategy / Percent Enumeration": self.results["Percent Enumeration"],
                    f"Reasoning Strategy / Percent Tree Search": self.results["Percent Tree Search"],
                    f"Reasoning Strategy / Percent Backtracking": self.results["Percent Backtracking"],
                    f"Reasoning Strategy / Percent Self Correction": self.results["Percent Self Correction"],
                    f"Reasoning Strategy / Percent Subgoal Setting": self.results["Percent Subgoal Setting"],
                    f"Reasoning Strategy / Percent Verification": self.results["Percent Verification"],
                    f"Reasoning Strategy / Percent Reprompts": self.results["Percent Reprompts"],
                })
        
        elif self.task_type == "reasoning_quality":
            self.results['Avg. Reasoning Efficacy'] = self._safe_div(self.results['Sum: Reasoning Efficacy'], self.results['Total Responses Parsed'])
            self.results['Avg. Reasoning Efficiency'] = self._safe_div(self.results['Sum: Reasoning Efficiency'], self.results['Total Responses Parsed'])
            self.results['Avg. Reasoning Faithfulness'] = self._safe_div(self.results['Sum: Reasoning Faithfulness'], self.results['Total Responses Parsed'])
            reasoning_score_all = self.results['Sum: Reasoning Efficacy'] + self.results['Sum: Reasoning Efficiency'] + self.results['Sum: Reasoning Faithfulness']
            self.results['Avg. Reasoning All'] = self._safe_div(reasoning_score_all, self.results['Total Responses Parsed'])
            self.results['Percent Reprompts'] = self._safe_div(self.results['Error: Reprompt'], self.results['Total Responses Parsed'])
            
            if self.wandb_run:
                self.wandb_run.log({
                    f"Reasoning Quality / Avg. Reasoning Efficacy": self.results["Avg. Reasoning Efficacy"],
                    f"Reasoning Quality / Avg. Reasoning Efficiency": self.results["Avg. Reasoning Efficiency"],
                    f"Reasoning Quality / Avg. Reasoning Faithfulness": self.results["Avg. Reasoning Faithfulness"],
                    f"Reasoning Quality / Avg. Reasoning All": self.results["Avg. Reasoning All"],
                    f"Reasoning Quality / Percent Reprompts": self.results["Percent Reprompts"],
                })

        return self.results


    # =================================================
    # Internal helper functions
    # =================================================
    def _instantiate_dict(self):
        if self.task_type == "hallucination":
            return {
                "Filename": self.filename,
                "Total Responses Parsed": 0,
                "Count: Moves Checked":  0,
                "Count: Moves Correct":  0,
                "Count: Pieces Checked": 0,
                "Count: Pieces Correct": 0,
                "Count: Hallucinations": 0,
                "Error: Reprompt": 0,
                "Error: Other": 0
            }
        elif self.task_type == "reasoning_strategy":
            return {
                "Filename": self.filename,
                "Total Responses Parsed": 0,
                "Count: Enumeration": 0,
                "Count: Tree Search": 0,
                "Count: Backtracking": 0,
                "Count: Self Correction": 0,
                "Count: Subgoal Setting": 0,
                "Count: Verification": 0,
                "Error: Reprompt": 0,
                "Error: Other": 0,
            }
        elif self.task_type == "reasoning_quality":
            return {
                "Filename": self.filename,
                "Total Responses Parsed": 0,
                "Sum: Reasoning Efficacy": 0,
                "Sum: Reasoning Efficiency": 0,
                "Sum: Reasoning Faithfulness": 0,
                "Error: Reprompt": 0,
                "Error: Other": 0,
            }
        else:
            raise ValueError(f"Undefined task type: {self.task_type}")

    def _safe_div(self, x, y, default=0): 
        return x / y if y else default
    


# =============================================
# Results Dict for Difficulty Parsing Cases
# =============================================
class DifficultyResultsDict():
    def __init__(self, task_type, filename, wandb_run):
        self.task_type = task_type
        self.filename = filename
        self.trimmed_filename = filename.split("_", 1)[0]
        self.wandb_run = wandb_run
        self.board_id_results = dict()
        self.results = self._instantiate_dict()

    def add_result(self, model_response, info):
        # First store metadata to track unique boards we encounter
        if info['board_id'] not in self.board_id_results:
            self.results['Count: Sample Questions'] += 1
            self.board_id_results[info['board_id']] = {
                'score_answers': [],
                'info': info
            }

        # Then need to get our answer / scores
        score = '<ERROR>'
        predicted_answer = '<ERROR>'
        try:
            self.results["Count: Total Generations"] += 1
            ground_truth = info['answer']

            if self.task_type == "choose_from_n":
                answer = ground_truth['answer']
                candidates = ground_truth['candidates']
                predicted_answer = coerce_response(extract_solution(model_response), self.task_type)

                if predicted_answer == answer:
                    score = 1
                else:
                    if predicted_answer in candidates:
                        score = 0
                    else:
                        raise IllegalMoveException("Predicted move is not in the provided moves.")
                self.results["Count: Legal Generations"] += 1
                self.results["Total: Cumulative Score"] += score
            
            elif self.task_type == 'produce_list':
                answer = ground_truth
                predicted_answer = coerce_response(extract_solution(model_response), self.task_type)

                # Compute correctness
                num_right = 0
                already_guessed = set()
                for move in predicted_answer:
                    if move in answer and move not in already_guessed:
                        already_guessed.add(move)
                        num_right += 1
                score = num_right / (len(answer) + len(predicted_answer) - num_right)
                self.results["Count: Legal Generations"] += 1
                self.results["Total: Cumulative Score"] += score
                
            elif self.task_type == 'predict_singlemove':
                answer = ground_truth
                predicted_answer = coerce_response(extract_solution(model_response), self.task_type)
                sorted_answers = sorted(answer.items(), key=lambda x: x[1])

                if predicted_answer in answer:
                    predicted_move_idx = next(i for i, (move, _) in enumerate(sorted_answers) if move == predicted_answer)
                    score = predicted_move_idx/len(sorted_answers)
                else:
                    raise IllegalMoveException("Predicted move is not in the legal moves.")
                self.results["Count: Legal Generations"] += 1
                self.results["Total: Cumulative Score"] += score
                
        # Exception handling to log various errors     
        except Exception as e:
            if isinstance(e, ParseException):
                self.results["Error: Parsing"] += 1
            elif isinstance(e, IllegalMoveException):
                self.results["Error: Illegal Move"] += 1
            else:
                self.results["Error: Other"] += 1
        
        # If we make it through without an error raised, can append
        self.board_id_results[info['board_id']]['score_answers'].append((score, predicted_answer))
    

    def get_final_dict(self):
        """ Return finalized dict and log to wandb. """
        average_score_all = self._safe_div(self.results["Total: Cumulative Score"], self.results['Count: Total Generations'])
        average_score_legal = self._safe_div(self.results["Total: Cumulative Score"], self.results['Count: Legal Generations'])
        total_errors = self.results['Error: Illegal Move'] + self.results['Error: Parsing'] + self.results['Error: Other'] 
        error_rate = self._safe_div(total_errors, self.results['Count: Total Generations'])

        self.results['Avg. Score - All'] = average_score_all
        self.results['Avg. Score - Legal'] = average_score_legal
        self.results['Error Rate'] = error_rate
        
        if self.wandb_run:
            self.wandb_run.log({
                f"Test Difficulty / Avg. Score - All": self.results['Avg. Score - All'],
                f"Test Difficulty / Avg. Score - Legal": self.results['Avg. Score - Legal'],          
                f"Test Difficulty / Error Rate": self.results['Error Rate']                
                })

        return self.results


    # =================================================
    # Internal helper functions
    # =================================================
    def _instantiate_dict(self):
        return {
            "Filename": self.filename,
            "Count: Sample Questions": 0,
            "Count: Total Generations": 0,
            "Count: Legal Generations": 0,
            "Total: Cumulative Score": 0,
            "Error: Illegal Move": 0,
            "Error: Parsing": 0,
            "Error: Other": 0
        }

    def _safe_div(self, x, y, default=0): 
        return x / y if y else default