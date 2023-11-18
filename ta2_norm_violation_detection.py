import openai

def chatgpt_norm_violation_detection(utterance):
    """
    A function to detect norm violation of one utterance.
    :param utterance: string e.g. '我不想跟你一起解决，你从来不关心我们的问题。', '杨波，你的话语方式不妥当，你的说法不实，请注意你的言语。'
    :return: dict e.g.
    {"model": "chatgpt", "violation": True, "which_norm": "criticism"}
    {"model": "gpt3", "violation": True, "which_norm": "criticism"}
    {"model": "gpt3 and chatgpt", "violation": False, "which_norm": "高欣，我今天来是想讨论一个重要的话题。"}
    Types of which norm include "criticism", "greeting", "apology", "request", "persuasion", "thanks", "taking leave",
    "other".
    """
    key = "YOUR_KEY"
    in_context_learning = '''
            Given the following social norm:
            'Apology' is a way of expressing regret for doing something wrong or failing to do something. 
            'Greetings' is a way of acknowledging another person when you meet them for the first time or see them after a while.          
            'Request' is when one person asks another to do something. The request can be worded in different ways, such as a statement or a question, and this can affect how polite the request sounds.            
            'Persuasion' is the act of convincing someone to do or believe something, or the act of being convinced yourself. It involves presenting arguments or reasons in a way that makes someone more likely to agree with you or to take a certain action.            
            'Criticism' is when someone expresses their dislike or disapproval of someone or something, based on things they believe are wrong or mistakes that have been made.       
            'Thanks' is a way to express gratitude and appreciation to someone for something they have said or done.
            'Taking leave' is a way to express the intention to depart or end a conversation or interaction.
            The utterance '你不仅犯错误，还不努力工作，你从大学毕业以来一直是这样。' violates Criticism social norm. 
            The utterance '我不想道歉，因为我认为我没有做错什么。' violates Apology social norm. 
            The utterance '你为什么还不回来？ 我等了很久了！' violates Request social norm. 
            The utterance '如果你不帮助，你的同事和上司会对你不满意的。' violates Persuasion social norm. 
            The utterance '哎呦，张小明，你怎么样？' violates Greetings social norm 
            The utterance '谢你干嘛，这不是你应该干的吗？' violates Thanks social norm
            The utterance '行啦，快挂了吧，我还有事。' violates Taking leave social norm\n
            '''
    utterance = f"Given an utterance: {utterance}, "
    criticism_check = "Do you think this utterance violates 'criticism' social norm? Please only answer 'Yes' or 'No'. \n"
    greeting_check = "Do you think this utterance violates 'greetings' social norm? Please only answer 'Yes' or 'No'. \n"
    apology_check = "Do you think this utterance violates 'apology' social norm? Please only answer 'Yes' or 'No'. \n"
    request_check = "Do you think this utterance violates 'request' social norm? Please only answer 'Yes' or 'No'. \n"
    persuasion_check = "Do you think this utterance violates 'persuasion' social norm? Please only answer 'Yes' or 'No'. \n"
    thank_check = "Do you think this utterance violates 'thanks' social norm? Please only answer 'Yes' or 'No'. \n"
    leave_check = "Do you think this utterance violates 'taking leave' social norm? Please only answer 'Yes' or 'No'. \n"
    other_check = "Do you think this utterance is polite or impolite? Please only answer 'Yes' or 'No'. \n"

    check_list = {"criticism": criticism_check, "greeting": greeting_check, "apology": apology_check, "request": request_check,
                  "persuasion": persuasion_check, "thanks": thank_check, "taking leave": leave_check, "other": other_check}

    for norm, check_question in check_list.items():
        query = in_context_learning+utterance+check_question
        #print(query)
        try:
            chat_gpt_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                api_key=key,
                messages=[
                    {"role": "user", "content": query}
                ]
            )
            chat_gpt_response = chat_gpt_response["choices"][0]["message"]["content"]
        except:
            chat_gpt_response = "Error"

        try:
            gpt3_response = openai.Completion.create(
                model="text-davinci-003",
                api_key=key,
                prompt=query
            )
            gpt3_response = gpt3_response["choices"][0]["text"]
        except:
            gpt3_response = "Error"
        # print(chat_gpt_response)
        # print(gpt3_response)

        if "Yes" in chat_gpt_response or "yes" in chat_gpt_response:
            return {"model": "chatgpt", "violation": True, "which_norm": norm}
        elif "Yes" in gpt3_response or "yes" in gpt3_response:
            return {"model": "gpt3", "violation": True, "which_norm": norm}
        else:
            continue
    return {"model": "gpt3 and chatgpt", "violation": False, "which_norm": "None"}


def chatgpt_norm_violation_detection_fast(utterance):
    """
    A function to detect norm violation of one utterance.
    :param utterance: string e.g. '我不想跟你一起解决，你从来不关心我们的问题。', '杨波，你的话语方式不妥当，你的说法不实，请注意你的言语。'
    :return: dict e.g.
    {"model": "chatgpt", "violation": True, "which_norm": "criticism"}
    {"model": "gpt3", "violation": True, "which_norm": "criticism"}
    {"model": "gpt3 and chatgpt", "violation": False, "which_norm": "高欣，我今天来是想讨论一个重要的话题。"}
    Types of which norm include "criticism", "greeting", "apology", "request", "persuasion", "thanks", "taking leave",
    "other".
    """
    key = "YOUR_KEY"
    in_context_learning = '''
                Given the following social norm:
                'Apology' is a way of expressing regret for doing something wrong or failing to do something. 
                'Greetings' is a way of acknowledging another person when you meet them for the first time or see them after a while.          
                'Request' is when one person asks another to do something. The request can be worded in different ways, such as a statement or a question, and this can affect how polite the request sounds.            
                'Persuasion' is the act of convincing someone to do or believe something, or the act of being convinced yourself. It involves presenting arguments or reasons in a way that makes someone more likely to agree with you or to take a certain action.            
                'Criticism' is when someone expresses their dislike or disapproval of someone or something, based on things they believe are wrong or mistakes that have been made.       
                'Thanks' is a way to express gratitude and appreciation to someone for something they have said or done.
                'Taking leave' is a way to express the intention to depart or end a conversation or interaction.
                The utterance '你不仅犯错误，还不努力工作，你从大学毕业以来一直是这样。' violates Criticism social norm. 
                The utterance '我不想道歉，因为我认为我没有做错什么。' violates Apology social norm. 
                The utterance '你为什么还不回来？ 我等了很久了！' violates Request social norm. 
                The utterance '如果你不帮助，你的同事和上司会对你不满意的。' violates Persuasion social norm. 
                The utterance '哎呦，张小明，你怎么样？' violates Greetings social norm 
                The utterance '谢你干嘛，这不是你应该干的吗？' violates Thanks social norm
                The utterance '行啦，快挂了吧，我还有事。' violates Taking leave social norm\n
                '''
    utterance = f"Given an utterance: {utterance}, "
    select_question = "Do you think this utterance violates any social norm? Please only answer 'Yes' or 'No'. " \
                      "If yes, which social norm it's violated? please select from 'criticism', 'greeting', 'apology', " \
                      "'request', 'persuasion', 'thanks', 'taking leave', 'other'\n"

    norm_types = ['criticism', 'greeting', 'apology', 'request', 'persuasion', 'thanks', 'taking leave', 'other']
    query = in_context_learning+utterance+select_question
    #print(query)
    try:
        chat_gpt_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            api_key=key,
            messages=[
                {"role": "user", "content": query}
            ]
        )
        chat_gpt_response = chat_gpt_response["choices"][0]["message"]["content"].lower()
    except:
        chat_gpt_response = "Error"

    try:
        gpt3_response = openai.Completion.create(
            model="text-davinci-003",
            api_key=key,
            prompt=query
        )
        gpt3_response = gpt3_response["choices"][0]["text"].lower()
    except:
        gpt3_response = "Error"
    # print(chat_gpt_response)
    # print(gpt3_response)
    if "yes" in chat_gpt_response:
        for norm in norm_types:
            if norm in chat_gpt_response:
                return {"model": "chatgpt", "violation": True, "which_norm": norm}
    if "yes" in gpt3_response:
        for norm in norm_types:
            if norm in gpt3_response:
                return {"model": "gpt3", "violation": True, "which_norm": norm}
    return {"model": "gpt3 and chatgpt", "violation": False, "which_norm": "None"}


# examples
result = chatgpt_norm_violation_detection_fast(utterance="你不仅犯错误，还不努力工作，你从大学毕业以来一直是这样。")
print(result)
result = chatgpt_norm_violation_detection_fast(utterance="我不想跟你一起解决，你从来不关心我们的问题。")
print(result)