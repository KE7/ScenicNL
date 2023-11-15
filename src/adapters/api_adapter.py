
"""
Scenic3 Mini API Helper Class
@TODO: [urgent] Make sure that code rewritten by the oneline helper is consistent with the rest of the code!
General Notes:
Not adding function types yet since the LLM output is not consistent and the try/catches currently handle edge* cases.
@TODO: Devan add try / except blocks inside methods to avoid errors thrown by string arguments.
(For context, calling ```exec()``` on all lines of LLM output throws some errors
 that can be caught by calling the str() command on function calling inputs or adding quotes to args. )
@TODO: Devan create a more extensive API that can fully express Scenic programs.
(All TODOs for Devan but anyone with extra time welcome to look over.)
@TODO: Devan make the API usage 1. more closely resemble UCLID5 paper 2. find better indentation solution
"""


class Scenic3:

    def __init__(self):
        self.code = []
        self._indent = ' ' * 4

    def set_map(self, map_name:any, indent=0):
        indent_str = self._indent * indent
        try:
            self.code.append(f"{indent_str}param map = localPath('{map_name}')")
        except:
            self.code.append(f"{str(indent_str)}param map = localPath('{map_name}')")

    def set_model(self, model_name:any, indent=0):
        indent_str = self._indent * indent
        try: self.code.append(f"{indent_str}model {model_name}")
        except: self.code.append(f"{(indent_str)}model {str(model_name)}")

    def define_constant(self, name:any, value:any, indent=0):
        indent_str = self._indent * indent
        try: self.code.append(f"{indent_str}{name} = {value}")
        except: self.code.append(f"{indent_str}{str(name)} = {str(value)}")

    def define_behavior(self, name:any, indent=0, **kwargs):
        indent_str = self._indent * indent
        kwargs_str = ', '.join(f'{str(k)}={str(v)}' for k, v in kwargs.items())
        try: self.code.append(f"{indent_str}behavior {name}({kwargs_str}):")
        except: self.code.append(f"{indent_str}behavior {str(name)}({kwargs_str}):")

    def do(self, behavior_name:any, indent=0, **kwargs):
        indent_str = self._indent * indent
        # behavior_name = str(behavior_name)
        kwargs_str = ', '.join(f'{str(k)}={str(v)}' for k, v in kwargs.items())
        try: self.code.append(f"{indent_str}do {behavior_name}({kwargs_str})")
        except: self.code.append(f"{indent_str}do {str(behavior_name)}({kwargs_str})")

    def do_while(self, behavior_name:any, var_name:any, condition:any, indent=0):
        indent_str = self._indent * indent
        # behavior_name:any, var_name:any, condition = str(behavior_name), str(var_name), str(condition)
        try: self.code.append(f"{indent_str}do {behavior_name}({var_name}) while {condition}")
        except: self.code.append(f"{indent_str}do {str(behavior_name)}({str(var_name)}) while {condition}")

    def do_until(self, behavior_name:any, var_name:any, condition:any, indent=0):
        indent_str = self._indent * indent
        # behavior_name:any, var_name = str(behavior_name), str(var_name)
        try: self.code.append(f"{indent_str}do {behavior_name}({var_name}) until {condition}")
        except: self.code.append(f"{indent_str}do {str(behavior_name)}({var_name}) until {condition}")

    def try_except(self, try_behavior, except_behavior, indent=0):
        indent_str = self._indent * indent
        try_behavior, except_behavior = str(try_behavior), str(except_behavior)
        self.code.append(f"{indent_str}try:")
        try: self.code.append(f"{indent_str+self._indent}do {try_behavior}")
        except: self.code.append(f"{indent_str+self._indent}do {str(try_behavior)}")
        self.code.append(f"{indent_str}except:")
        try: self.code.append(f"{indent_str+self._indent}do {except_behavior}")
        except: self.code.append(f"{indent_str+self._indent}do {str(except_behavior)}")

    def interrupt(self, condition:any, indent=0, *args):
        indent_str = self._indent * indent
        args_str = ', '.join(str(arg) for arg in args)
        try: self.code.append(f"{indent_str}interrupt when {condition}({args_str}):")
        except: self.code.append(f"{indent_str}interrupt when {str(condition)}({args_str}):")

    def take(self, action_name:any, indent=0, **params):
        indent_str = self._indent * indent
        params_str = ', '.join(f'{str(k)}={str(v)}' for k, v in params.items())
        try: self.code.append(f"{indent_str}take {action_name}({params_str})")
        except: self.code.append(f"{indent_str}take {str(action_name)}({params_str})")

    def new(self, var_name:any, obj_type:any, at=None, indent=0, **kwargs):
        indent_str = self._indent * indent
        try: new_line = f"{indent_str}{var_name} = new {obj_type.capitalize()}"
        except: new_line = f"{indent_str}{str(var_name)} = new {str(obj_type).capitalize()}"
        
        if at:
            new_line += f" at {at}"
        
        for k, v in kwargs.items():
            new_line += f", {str(k)}={str(v)}"

        self.code.append(new_line)

    def spatial_relation(self, obj1:any, keyword:any, obj2:any, distance=None, indent=0):
        indent_str = self._indent * indent

        if distance:
            try: self.code.append(f"{indent_str}{obj1} {keyword} {obj2} for {distance}")
            except: self.code.append(f"{indent_str}{str(obj1)} {str(keyword)} {str(obj2)} for {str(distance)}")
        else:
            try: self.code.append(f"{indent_str}{obj1} {keyword} {obj2}")
            except: self.code.append(f"{indent_str}{str(obj1)} {str(keyword)} {str(obj2)}")

    def Uniform(self, seq:any, indent=0):
        indent_str = self._indent * indent
        return f"{indent_str}Uniform({seq})"

    def Range(self, start:any, end:any, indent=0):
        indent_str = self._indent * indent
        return f"{indent_str}Range({start}, {end})"

    def require(self, condition:any, indent=0):
        indent_str = self._indent * indent
        try: self.code.append(f"{indent_str}require {condition}")
        except: self.code.append(f"{indent_str}require {str(condition)}")

    def terminate(self, condition:any, indent=0):
        indent_str = self._indent * indent
        try: self.code.append(f"{indent_str}terminate when {condition}")
        except: self.code.append(f"{indent_str}terminate when {str(condition)}")

    def add_code(self, lines:list[str]):
        # only use on API failures
        for line in lines:
            self.code.append(line.strip('\n'))

    def get_code(self):
        return "\n".join(self.code).strip()
