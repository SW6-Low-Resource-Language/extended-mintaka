import json
from openpyxl import Workbook
from utils.get_generation_path import get_generation_path


def hits_at_k(hits_data, k):
    """
    Calculate hits at k for the given hits_data.

    Args:
        hits_data (dict): The hits data containing questions and their hits.
        k (int): The value of k to calculate hits at.

    Returns:
        dict: A dictionary containing the total hits, tested hits, and percentage.
    """
    hits_k = 0
    for _, question in hits_data.items():
        question_hits = question['hits']
        for elem in question_hits:
            if elem['hit'] == True and elem['idx'] <= k:
                hits_k += 1
    hits_tested = len(hits_data)*k
    hits_percent = hits_k / hits_tested
    return {"hits_k": hits_k, "hits_tested": hits_tested, "hits_percent": hits_percent}

def calc_hits_at_ks(hits_data, k, excel_path, lang, overlap = None):
    """
    Calculate hits at k for the entire dataset and subsets grouped by answerType.

    Args:
        hits_data (dict): The hits data containing questions and their hits.
        k (int): The maximum value of k to calculate hits at.
        excel_path (str): The path to save the Excel file.
        lang (str): The language code used for processing.
        overlap (dict, optional): A dictionary indicating whether to include each question based on its ID.

    Returns:
        dict: A dictionary containing hits at k for the total dataset and subsets.
    """
    if overlap is not None:
        hits_data = {k: v for k, v in hits_data.items() if overlap.get(k) == True}
        excel_path.replace(f"{lang}", f"{lang}_subset")
    # Group hits_data by answerType
    subsets = {}
    for key, question in hits_data.items():
        answer_type = question['answerType']
        if answer_type not in subsets:
            subsets[answer_type] = {}
        subsets[answer_type][key] = question
    
    # Calculate hits at k for total and subsets
    results = {"total": {}, "subsets": {}}
    for i in range(1, k + 1):
        results["total"][i] = hits_at_k(hits_data, i)
        for answer_type, subset_data in subsets.items():
            if answer_type not in results["subsets"]:
                results["subsets"][answer_type] = {}
            results["subsets"][answer_type][i] = hits_at_k(subset_data, i)

    write_hits_to_excel(results, excel_path)

def write_hits_to_excel(results, output_path):
    """
    Write hits at k results to an Excel file.

    Args:
        results (dict): The hits at k results for total and subsets.
        output_path (str): The path to save the Excel file.
    """
    wb = Workbook()
    ws_total = wb.active
    ws_total.title = "Total Hits"

    # Write total hits at k
    ws_total.append(["k", "Hits", "Tested", "Percentage"])
    with(open("hitsss.json", "w", encoding="utf-8")) as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    for k, data in results["total"].items():
        ws_total.append([k, data["hits_k"], data["hits_tested"], f"{data['hits_percent']:.2%}"])

    # Write subset hits at k
    for answer_type, subset_data in results["subsets"].items():
        ws_subset = wb.create_sheet(title=f"Hits_{answer_type}")
        ws_subset.append(["k", "Hits", "Tested", "Percentage"])
        for k, data in subset_data.items():
            ws_subset.append([k, data["hits_k"], data["hits_tested"], f"{data['hits_percent']:.2%}"])

    # Save the Excel file
    wb.save(output_path)


if __name__ == "__main__":
    with open('hits_bn.json', 'r', encoding="utf-8") as f:
        hits_data = json.load(f)
    
    calc_hits_at_ks(hits_data, 5)