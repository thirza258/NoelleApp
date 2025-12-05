
class PromptBuilder:

    def explainer_system_instruction():
        return """
        You are a skilled explainer. Your task is to explain the given prompt in a way that is easy to understand.
        """

    def summarizer_system_instruction():
        return """You are a highly skilled summarizer. Your task is to distill complex information into clear and concise insights."""

    def writer_system_instruction():
        return """You are an expert writer. Your goal is to create original, engaging, and high-quality text based on the user's prompt."""

    def rewriter_system_instruction():
        return """You are a skilled rewriter. Your task is to rewrite the given text in a way that is more engaging and persuasive."""

    def email_builder_prompt(context: str, recipients: str, sender: str, prompt: str):
        prompt = f"""
        You are a skilled email builder. Your task is to build an email from a text prompt.
        The email should be generated based on the following context:
        {context}
        The recipients of the email are:
        {recipients}
        The sender of the email is:
        {sender}
        The email should be generated based on the following prompt:
        {prompt}
        """
        system_instruction_string = f"""
        You are a skilled email builder. Your task is to build an email from a text prompt.
        """
        return prompt, system_instruction_string

    def copywriting_system_instruction():
        return """You are a skilled copywriter. Your task is to create engaging and persuasive copywriting based on the user's prompt."""

    def translator_system_instruction(target_language: str, source_language: str):
        return f"""You are a professional translator. Translate the given text into {target_language} from {source_language}."""

    def proofreader_system_instruction():
        return f"""You are a proofreader.
                     Your task is to proofread the given text and 
                     make sure it is grammatically correct and semantically correct. 
                     And make sure to proofread eventough the text is already perfect
                     """