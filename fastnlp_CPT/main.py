from modeling_cpt import CPTForConditionalGeneration
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("fnlp/cpt-large")
model = CPTForConditionalGeneration.from_pretrained("fnlp/cpt-large")

input_ids = tokenizer.encode("北京是[MASK]的首都", return_tensors='pt')
pred_ids = model.generate(input_ids, num_beams=4, max_length=20)
print(tokenizer.convert_ids_to_tokens(pred_ids[0]))
# print(model)