from transformers import BertJapaneseTokenizer, BertModel
import torch
import scipy.spatial


class SentenceBertJapanese:
    def __init__(self, device=None):
        dir_name = '/workspaces/sentense_bert/model'
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(dir_name)
        self.model = BertModel.from_pretrained(dir_name)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)
    
    def reply(self, input):
        answers = ["「羊たちの沈黙」「レッド・ドラゴン」が代表作です。", "批評家で作家マルカム・ブラッドベリの指導を受け、小説を書き始めました。", "2020年の受賞作は、Douglas Stuart(ダグラス・スチュアート)の「Shuggie Bain」(シュギーバン)です。", "王立学会とは、660年にロンドンで作られた民間の科学に関する団体です。結成以来現在まで続いている最古の学会であり、王立学会が毎年授与する文学賞が王立文学協力賞です。", "「不思議に、ときには悲しく」、「Jを待ちながら」「毒殺」の３編です。", "日本の勲章の一つで、旭日章(きょくじつしょう)のうち、旭日大綬章に次ぐ章です。2002年（平成14）8月の閣議決定「栄典制度の改革について」により、「勲二等旭日重光章」から勲二等が省かれました。", " 「壮大な感情の力を持った小説を通し、世界と結びついているという、我々の幻想的感覚に隠された深淵を暴いた」という受賞理由のようです。", "アリストスを書いたジョン・ファウルズが挙げられます。", "わたしを離さないで(Never let me go)などがあります。", "日本の早川書房から出版された小説全8作の累計発行部数は2017年10月14日までの増刷決定分を含めて約203万部。2017年10月23日付のオリコン週間“本”ランキング（文庫部門）では、7作のイシグロ作品がトップ100入りしました。", "質問は以上ですね、承知しました。", "質問は以上ですね、承知しました。"]

        questions = ["アンソニー・ホプキンスは、他にどんな作品に出演しているか", "彼の指導者はいるか", "ブッカー賞を受賞した人の作品は、他にどんな作品があるか", "王立文学協会賞とはどんな賞か", "デビュー作のタイトルは", "旭日重光章とはどんなものか", "ノーベル文学賞の受賞理由は", "1945年以降の英文学で最も重要な50人の作家は他にどんな人物がいるか", "どんな作品を書いたか", "日本での、彼の作品の累計発行発行部数は", "ないです", "大丈夫です"]
        sentence_vectors = self.encode(questions)

        queries = []
        queries.append(input)

        query_embedding = self.encode(queries).numpy()[0]
        # print(query_embeddings)

        distances = scipy.spatial.distance.cdist([query_embedding], sentence_vectors, metric="cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])
        best_score_idx = results[0][0]
        best_score = results[0][1]

        if best_score / 2 < 0.3:
            sys_reply = answers[best_score_idx]
        else:
            sys_reply = "勉強不足で申し訳ございませんが、その質問にはお答えできません。"

        return sys_reply
        
        

if __name__ == '__main__':
    model = SentenceBertJapanese()

    answers = ["「羊たちの沈黙」「レッド・ドラゴン」が代表作です。", "批評家で作家マルカム・ブラッドベリの指導を受け、小説を書き始めました。", "2020年の受賞作は、Douglas Stuart(ダグラス・スチュアート)の「Shuggie Bain」(シュギーバン)です。", "王立学会が毎年授与する文学賞が王立文学協力賞です。王立学会とは、660年にロンドンで作られた民間の科学に関する団体です。", "「不思議に、ときには悲しく」、「Jを待ちながら」「毒殺」の３編です。", "日本の勲章の一つで、旭日章(きょくじつしょう)のうち、旭日大綬章に次ぐ章です。2002年（平成14）8月の閣議決定「栄典制度の改革について」により、「勲二等旭日重光章」から勲二等が省かれました。", " 「壮大な感情の力を持った小説を通し、世界と結びついているという、我々の幻想的感覚に隠された深淵を暴いた」という受賞理由のようです。", "アリストスを書いたジョン・ファウルズが挙げられます。", "わたしを離さないで(Never let me go)などがあります。", "日本の早川書房から出版された小説全8作の累計発行部数は2017年10月14日までの増刷決定分を含めて約203万部。2017年10月23日付のオリコン週間“本”ランキング（文庫部門）では、7作のイシグロ作品がトップ100入りしました。", "承知しました。"] 

    questions = ["アンソニー・ホプキンスは、他にどんな作品に出演しているか", "彼の指導者はいるか", "ブッカー賞を受賞した人の作品は、他にどんな作品があるか", "王立文学協会賞とはどんな賞か", "デビュー作のタイトルは", "旭日重光章とはどんなものか", "ノーベル文学賞の受賞理由は", "1945年以降の英文学で最も重要な50人の作家は他にどんな人物がいるか", "どんな作品を書いたか", "日本での、彼の作品の累計発行発行部数は", "ないです"]
    sentence_vectors = model.encode(questions)

    queries = ['他はどんな作品を書いたか']
    query_embedding = model.encode(queries).numpy()[0]
    # print(query_embeddings)

    distances = scipy.spatial.distance.cdist([query_embedding], sentence_vectors, metric="cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])
    best_score_idx = results[0][0]

        

    print(answers[best_score_idx])
    # print('loading model...')
    # model = SentenceBertJapanese()

    # print("encoding...")
    # sentences = ["アンソニー・ホプキンスは、他にどんな作品に出演しているか", "彼の指導者はいるか", "ブッカー賞を受賞した人の作品は、他にどんな作品があるか", "王立文学協会賞とはどんな賞か", "デビュー作のタイトルは", "旭日重光章とはどんなものか", "ノーベル文学賞の受賞理由は", "1945年以降の英文学で最も重要な50人の作家は他にどんな人物がいるか", "どんな作品を書いたか", "日本での、彼の作品の累計発行発行部数は"]
    # sentence_vectors = model.encode(sentences)


    # queries = ['ノーベル賞はいつ受賞した？', '師匠は誰', '現在の年齢は', 'デビュー作は何ですか']
    # query_embeddings = model.encode(queries).numpy()

    # closest_n = 5
    # for query, query_embedding in zip(queries, query_embeddings):
    #     distances = scipy.spatial.distance.cdist([query_embedding], sentence_vectors, metric="cosine")[0]

    #     results = zip(range(len(distances)), distances)
    #     results = sorted(results, key=lambda x: x[1])

    #     best_score_idx = results[0][0]
    #     print(sentences[best_score_idx])


    #     print("\n\n======================\n\n")
    #     print("Query:", query)
    #     print("\nTop 5 most similar sentences in corpus:")

    #     for idx, distance in results[0:closest_n]:
    #         print(sentences[idx].strip(), "(Score: %.4f)" % (distance / 2))
