import random, re, os
from comfy_api.latest import io

from ..config import config
from .utils import get_node_id, get_node_category

class Processor:
    def __init__(self):
        # {one|two} ; {2$$one|two|tree} ; {1-2$$, $$one|two|three|four}
        # min_count, max_count, delimiter, options
        self.block_pattern = re.compile(r'\{(?:(?P<min_count>\d+)-?(?P<max_count>\d+)?\$\$(?:(?P<delimiter>[^{}]+)\$\$)?)?(?P<options>[^\$][^{}]+)\}')
        
        # __wildcards__ ; __path/wildcards__ ;
        self.wildcard_pattern = re.compile(r'(?:__(?P<key>[\w\/_\-. \*]+?)__)')

        # variable declaration: {!name=value!}
        self.variable_def_pattern = re.compile(r'\{!(?P<name>\w+)=(?P<value>.*?)!\}(?:[\s]*[\n])?')
        
        # variable: {$name}
        self.variable_pattern = re.compile(r'\{\$(?P<name>\w+)\}')

        # inline comment: /* ... */
        self.comment_pattern = re.compile(r'\/\*.*?\*\/', re.S)   

        # lora
        self.lora_pattern = re.compile(r'<lora:([^:]+):([^>]+)>')   

        self.wildcards_dict = None
        self.wildcards_cache = True
        self.wildcards_cleanup = True
        self.variables = {}

        self.load_settings()

    def load_settings(self):
        config.update()
        self.wildcards_path = config.wildcards.path 
        self.delimiter = config.wildcards.delimiter
        self.max_iterations = max(config.wildcards.max_iterations, 100) 
        self.wildcards_cache = config.wildcards.cache 
        self.wildcards_cleanup = config.wildcards.cleanup 

    def get_wildcards(self, key):
        if self.wildcards_dict is None:
            self.wildcards_dict = self.load_wildcards(self.wildcards_path) 

        key = key.lower()

        if '*' in key:
            # convert wildcard to regex pattern
            pattern = re.escape(key).replace(r'\*', '.*')
            matches = [k for k in self.wildcards_dict if re.fullmatch(pattern, k)]
            return [f"__{k}__" for k in matches]
        else:
            options = self.wildcards_dict[key] if key in self.wildcards_dict else []
            return options.copy()
    
    def remove_comments(self, text):
        # filter comment lines out
        text = '\n'.join(filter(lambda x: not x.startswith('#'), text.split('\n')))
        # remove inline comments
        return self.comment_pattern.sub('', text)
    
    def cleanup(self, text):  
        text = self.remove_comments(text)

        if not self.wildcards_cleanup:
            return text
              
        return self.lora_pattern.sub('', text)


    def execute(self, text: str, seed: int) -> str:
        self.load_settings()

        self.rng = random.Random(seed)

        processed_text = text
        processed_text = self.cleanup(processed_text)
        processed_text = self.variable_def_pattern.sub(self.replace_variable_def, processed_text)

        self.variables = {key: self.expand_text(value) for key, value in self.variables.items()}
        processed_text = self.variable_pattern.sub(self.replace_variable, processed_text)

        processed_text = self.expand_text(processed_text)               

        # delete the wildcards list    
        if not self.wildcards_cache:
            self.wildcards_dict = None      

        return processed_text
    
    def expand_text(self, text):
        for i in range(self.max_iterations): 
            old_text = text           
            text = self.block_pattern.sub(self.expand_block, text)
            text = self.wildcard_pattern.sub(self.replace_wildcards, text)            
            if text == old_text:
                return text

        return text
     
    
    def get_match_group(self, match, name, default_value=None): 
        if name not in match.groupdict():
            return default_value
        
        return match.group(name) or default_value
    
    def parse_block(self, match):
        min_count = int(self.get_match_group(match, 'min_count', 1)) 
        max_count = int(self.get_match_group(match, 'max_count', min_count))
        delimiter = self.get_match_group(match, 'delimiter', self.delimiter)
        options_str = self.get_match_group(match, 'options', '')

        options = options_str.split('|')    
        count = self.rng.randint(min_count, max_count)

        return count, options, delimiter
    
    def parse_wildcard(self, match):
        key = self.get_match_group(match, 'key')

        return key
    
    def replace_variable(self, match):
        name = self.get_match_group(match, 'name')

        return self.variables.get(name, match.group())

    def replace_variable_def(self, match):
        name = self.get_match_group(match, 'name')
        value = self.get_match_group(match, 'value')

        self.variables[name] = value

        return ''

    def expand_block(self, match):
        count, options, delimiter = self.parse_block(match)            

        weights = []
        weighted_options = []

        # check if the option is a wildcard
        if count > 1 and len(options) == 1: 
            m = self.wildcard_pattern.match(options[0])
            if m:
                wildcards_key = self.get_match_group(m, 'key')
                options = self.get_wildcards(wildcards_key)
                if not len(options):
                    return f"__{wildcards_key}__"

        for option in options:
            if '::' in option:
                weight, opt = option.split('::', 1)                
                try:
                    weights.append(float(weight))
                except ValueError:
                    weights.append(1.0)

                weighted_options.append(opt)
            else:
                weights.append(1)
                weighted_options.append(option)

        selected_options = self.weighted_select(count, weighted_options, weights)

        return delimiter.join(selected_options)
    
    def replace_wildcards(self, match):
        key = self.parse_wildcard(match)
        options = self.get_wildcards(key)
        
        selected = self.weighted_select(1, options)

        return selected[0] if len(options) else f"__{key}__"
        
    # select options without repeat
    def weighted_select(self, count, options, weights=None):
        count = min(count, len(options))
        options = options.copy()
        weights = weights.copy() if weights is not None else None
        selected = []

        for _ in range(count):
            # select a random index
            idx = self.rng.choices(
                range(len(options)),
                weights=weights,
                k=1
            )[0]

            selected.append(options.pop(idx))
            if weights is not None:
                weights.pop(idx)

        return selected
    
    def load_wildcards(self, path):
        wildcard_dict = {}
        
        for root, dirs, files in os.walk(path):
            for filename in files:
                if filename.endswith('.txt'):
                    file_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(file_path, path)  # Get the relative path
                    key = os.path.splitext(relative_path)[0]  # Remove the .txt extension
                    
                    # Convert to lowercase and replace backslashes with forward slashes
                    key = key.replace(os.sep, '/').lower()
                    
                    with open(file_path, 'r', encoding="utf8") as file:
                        lines = [line.strip() for line in file if not line.startswith('#') and line.strip()]
                    
                    wildcard_dict[key] = lines
        
        return wildcard_dict
 
_processor = None
class WildcardProcessor(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=get_node_id("WildcardProcessor"),
            display_name="Wildcard Processor",
            category=get_node_category(),
            inputs=[    
                io.String.Input("text", multiline=True),        
                io.Int.Input("Seed", default=0, min=0),
            ],
            outputs=[
                io.String.Output(display_name="text"),
            ],
        ) 

    @classmethod
    def get_processor(cls):
        global _processor
        if _processor is None:
            _processor = Processor()
        return _processor

    @classmethod
    def execute(cls, text, Seed):                               
        text = cls.get_processor().execute(text, Seed)

        return io.NodeOutput(text)      