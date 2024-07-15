# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:34:42 2024

@author: Admin
"""

import re
import numpy as np
def extract_numbers_from_file(filename):
    # Regular expression pattern to match 3-letter code followed by a number
    pattern = r'\b[A-Za-z]{3}\s*:\s*(\d+(,\d+)*)\b'
    numbers = []

    with open(filename, 'r',encoding='utf-8') as file:
        for line in file:
            matches = re.findall(pattern, line)
            for match in matches:
                numbers.append(match[0].replace(',', ''))

    return numbers

def main():
    filename = 'extracted_info.txt'  # Update with your file path
    numbers = np.array(extract_numbers_from_file(filename))
    np.save('PopNumbers.npy',numbers)
    # Print the extracted numbers
    #for number in numbers:
        #print(number)

if __name__ == "__main__":
    main()
    
