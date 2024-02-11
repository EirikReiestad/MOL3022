import pandas as pd


def read_protein_file_as_pd(file_path):
    with open(file_path, "r") as file:
        # Read the entire contents of the file
        seq = ""
        str = ""

        sequences = []
        strings = []
        count = 0
        for line in file:
            if count < 8:
                pass
            count += 1

            line = line.strip()
            if line == "<>":
                seq = ""
                str = ""
            elif line == "<end>":
                sequences.append(seq)
                strings.append(str)
            else:
                letters = line.split(" ")
                if len(letters) == 2:
                    seq += letters[0]
                    str += letters[1] if letters[1] != "_" else "c"
        df = pd.DataFrame({"sequence": sequences, "string": strings})
        return df
