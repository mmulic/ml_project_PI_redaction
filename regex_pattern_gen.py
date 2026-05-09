import  pandas as  pd
import  random
import  re
from  collections import  defaultdict

# Load CSV
df =  pd.read_csv("group_training.csv")

# Sample 20 random rows
sample_df =  df.sample(n=20)

# Classes to generate regex for
classes_of_interest =  [
      "email", "tel", "ip", "date", "bod", "time", "postcode",
      "geocoord", "building", "secaddress", "passport", "username",
      "pass", "idcard", "driverlicense", "socialnumber"
]

# Collect examples
examples =  defaultdict(list)
for  idx, row in  sample_df.iterrows():
      text =  row['source_text']
      spans = eval(row['span_labels'])
      for  start, end, label in  spans:
            label =  label.lower()
            if  label in  classes_of_interest:
                  examples[label].append(text[start:end])

# Detect character type
def char_type(c):
      if  c.isdigit():
            return 'd'
      elif  c.isalpha():
            return 'a'
      elif  c.isspace():
            return 's'
      else:
            return  c

# Generate regex with fixed-length sequences when consistent
def generate_length_aware_regex(examples_list):
      if not  examples_list:
            return None
     
      patterns =  []

      for  ex in  examples_list:
            pattern = ''
            prev_type = None
            count = 0
            for  c in  ex + '\0':   # sentinel to flush last sequence
                  cur_type =  char_type(c)
                  if  cur_type ==  prev_type:
                        count += 1
                  else:
                        if  prev_type == 'd':
                              pattern += f'\\d' +  (f'{{{count}}}' if  count > 1 else '')
                        elif  prev_type == 'a':
                              pattern += f'[a-zA-Z]' +  (f'{{{count}}}' if  count > 1 else '')
                        elif  prev_type == 's':
                              pattern += r'\s+'   # whitespace can vary
                        elif  prev_type and  prev_type not in  ['d','a','s']:
                              # For common separators, make optional
                              if  prev_type in  ['-', '/', '.', ':', ',', '#', '@']:
                                    pattern +=  re.escape(prev_type) + '?'
                              else:
                                    pattern +=  re.escape(prev_type)
                        # reset for new type
                        prev_type =  cur_type
                        count = 1
            patterns.append(pattern)
     
      # Combine patterns into one regex with OR
      combined_pattern = '|'.join(list(set(patterns)))
      return  combined_pattern

# Generate regex per class
regex_patterns =  {}
for cls, ex_list in  examples.items():
      regex_patterns[cls] =  generate_length_aware_regex(ex_list)

# Print results
for cls, pattern in  regex_patterns.items():
      print(f"{cls}: {pattern}")