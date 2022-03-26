class MainBot:
    # コンストラクタ
    def __init__(self, system):
        self.system = system

    def run(self, input: str):
        # replyメソッドによりinputから発話を生成
        system_output = self.system.reply(input)
        return system_output