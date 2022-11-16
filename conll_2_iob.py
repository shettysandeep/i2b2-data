"""
Converting the i2b2 data into CONLL IOB format for NN experiments
__author__: Sandeep Shetty
__date__: Nov 3, 2022

"""
import os
import re
from os.path import basename, isfile, isdir, join
import pandas as pd


## TODO: Doesn't handle blank ast files well. Pandas related error.


class i2b2CoNLL:
    def __init__(self, txt_pth, ast_pth):
        self.txt_pth = txt_pth
        self.ast_pth = ast_pth
        if isdir(self.txt_pth) and isdir(self.ast_pth):
            self.doc_files = [
                join(self.txt_pth, file)
                for file in os.listdir(self.txt_pth)
                if file.endswith(".txt")
            ]
            self.lbl_files = [
                os.path.join(self.ast_pth, file)
                for file in os.listdir(self.ast_pth)
                if file.endswith(".ast")
            ]

    def collect_files(self):
        doc_files_list = [basename(f).split(".")[0] for f in self.doc_files]
        lbl_files_list = [basename(f).split(".")[0] for f in self.lbl_files]
        # Only if .txt has corresponding .ast files
        both_files = set(doc_files_list).intersection(lbl_files_list)
        doc_list = []
        lbl_list = []
        for files in both_files:  # self.doc_files:
            # base_name = os.path.basename(files).split(".")[0] + ".ast"
            txt_name_pth = join(self.txt_pth, files + ".txt")
            lbl_name_pth = join(self.ast_pth, files + ".ast")
            doc_list.append(txt_name_pth)
            lbl_list.append(lbl_name_pth)
        return doc_list, lbl_list

    def read_file(self, file_path):
        with open(file_path) as f:
            content = f.readlines()
        return content

    def prcss_txt_lbl(self, text_content, lbl_content):
        # processing raw texts
        txt_data = pd.DataFrame(text_content)
        txt_data["LineNo"] = txt_data.index + 1
        # processing the concept and assertions data
        lbl_data = pd.DataFrame(lbl_content)
        lbl_data.rename(columns={0: "label"}, inplace=True)
        # Extract line number, word index, etc
        lbl_tmp = lbl_data.label.str.extract(
            r"(?P<line1>\d+)\:(?P<startWord>\d+)\s?(?P<line2>\d+)\:(?P<endWord>\d+)"
        )
        # Extract problem and assertion
        lbl_data["problem"] = lbl_data.label.str.extract('t="([a-z]+)')
        lbl_data["assert"] = lbl_data.label.str.extract('a="([a-z]+)')
        lbl_data = lbl_data.merge(
            lbl_tmp, how="left", right_index=True, left_index=True
        )
        # Convert to integer
        convt_integer = {"line1": int, "startWord": int, "line2": int, "endWord": int}
        lbl_data = lbl_data.astype(convt_integer)

        # Combine the doc text with labels on line number
        comb_data = txt_data.merge(
            lbl_data, left_on="LineNo", right_on="line1", how="left"
        )
        comb_data.rename(columns={0: "newtext"}, inplace=True)
        return comb_data

    def doc_iob(self, txt_doc_pth, ast_doc_pth):
        base_file = os.path.basename(txt_doc_pth).split(".")[0]
        lbl_file = os.path.basename(ast_doc_pth).split(".")[0]
        if base_file == lbl_file:
            # pass
            print("Processing...", base_file)
        else:
            print("\n STOPPING--- \n.ast filename not same as .txt")
            exit()
        content = self.read_file(txt_doc_pth)
        lblcontent = self.read_file(ast_doc_pth)
        comb_data = self.prcss_txt_lbl(content, lblcontent)
        columns_data = ["Doc_Id", "Sent_id", "Token", "NER_Tag", "Assertion"]
        iob_data = pd.DataFrame(columns=[columns_data])
        for ind in comb_data.index:
            # newlist = []
            # this is an assumption to check on splitting
            txt = comb_data.loc[ind, "newtext"].split(" ")
            lineNo = comb_data.loc[ind, "line1"]
            startword = comb_data.loc[ind, "startWord"]
            endword = comb_data.loc[ind, "endWord"]
            pblm = comb_data.loc[ind, "problem"]
            ast = comb_data.loc[ind, "assert"]
            sent_id = comb_data.loc[ind, "LineNo"]
            if lineNo != lineNo:
                for item in txt:
                    iob_data.loc[len(iob_data)] = [base_file, sent_id, item, "O", ""]
            else:
                for inde, item in enumerate(txt):
                    if inde == startword:
                        iob_data.loc[len(iob_data)] = [
                            base_file,
                            sent_id,
                            item,
                            "B-PROBLEM",
                            ast,
                        ]
                    if (inde > startword) and (inde <= endword):
                        iob_data.loc[len(iob_data)] = [
                            base_file,
                            sent_id,
                            item,
                            "I-PROBLEM",
                            ast,
                        ]
            # Blank row after the end of line
            iob_data.loc[len(iob_data)] = [base_file, sent_id, "", "", ""]
        return iob_data

    def convert_to_conll(self):
        if os.path.isfile(self.txt_pth) and os.path.isfile(self.ast_pth):
            doc_output = self.doc_iob(self.txt_pth, self.ast_pth)
            return doc_output
        else:
            doc_files, lbl_files = self.collect_files()
            corpus_output = pd.DataFrame()
            for file in doc_files:
                base_file = os.path.basename(file).split(".")[0]
                lbl_file_name = base_file + ".ast"
                lbl_file = os.path.join(self.ast_pth, lbl_file_name)
                corp_out = self.doc_iob(file, lbl_file)
                corpus_output = pd.concat([corpus_output, corp_out], axis=0)
        return corpus_output


if __name__ == "__main__":

    TxtfileName = "../i2b2_2010/beth/txt/"  # record-105.txt"
    LblfileName = "../i2b2_2010/beth/ast/"  # record-105.ast"

    conll = i2b2CoNLL(TxtfileName, LblfileName)
    print(conll.ast_pth)
    output = conll.convert_to_conll()
    output.to_csv("../conll_i2b2.csv")
