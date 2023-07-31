# TODO:
# 1. include text into node
# 2. handle 'or'
# 3. handle NP+PP, such as "what color is the vehicle on the left", "what vehicle are the trees behind of", if no 'of' then don't insert verb
# 4. better answer insertion, such as "what color is the vehicle on the left" -> "the color of the vehicle on the left is purple", "what vehicle are the trees behind of" -> "the vehicle that the trees are behind of is truck"
from copy import deepcopy
from re import L
# from pattern.en import conjugate, PL, PRESENT
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer

import benepar, spacy
nlp = spacy.load('en_core_web_md')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

DEBUG = False

WORDLEVEL_TAGS = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WRB", ".", ","] + ["HYPH"]
VB_TAGS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'MD']
VB_WORDS = ('do', 'does', 'can', 'could', 'would', 'should', 'might', 'has', 'have', "'ve", 'is', "'s", 'are', "'re", 'was', 'were')
STOPWORDS = stopwords.words('english') + ["side", "maybe", "part", "half", "picture", "photo", "image"]
STOPWORDS.remove('no')

def get_parse_tree_for_batch(texts):
    """
    texts: a list of text
    """
    prepared_texts = []
    all_choices = []
    for text in texts:
        choices = []
        # prepare text before parsing
        if ' do you think' in text:
            text = text.replace(" do you think", '')
        if 'Do you think the' in text and ' or ' in text:
            text = text.replace("Do you think the", "The")
        if 'Do you' in text:
            text = text.replace("Do you", "Do I")
        if 'in this photo ' in text:
            text = text.replace(" in this photo ", ' ')
        if 'in this picture' in text:
            text = text.replace(" in this picture", '')
        if 'in this image' in text:
            text = text.replace(" in this image", '')
        if 'in this photograph' in text:
            text = text.replace(" in this photograph", '')
        if 'in the photo ' in text:
            text = text.replace(" in the photo ", ' ')
        if 'in the picture' in text:
            text = text.replace(" in the picture", '')
        if 'in the image' in text:
            text = text.replace(" in the image", '')
        if 'in the photograph' in text:
            text = text.replace(" in the photograph", '')
        if 'Of what material the' in text:
            text = text.replace("Of what material the", 'Of what material is the')
        if ',' in text:
            text, choice = text.split(",")
            text += "?"
            assert " or " in choice
            if "?" in choice:
                choice = choice.replace("?", '')
            choices = [c.strip().split() for c in choice.strip().split(" or ")]
        if text.startswith("In front of"):
            text = text.replace("In front of ", '')
            text = text[:-1] + " in front of?"
        if 'in front or behind' in text:
            # the parser can not correctly parse in front or behind
            text = text.replace("in front or behind", "a or b")
        if 'behind or in front of' in text:
            text = text.replace("behind or in front of", 'a or b')
        # save into a list
        prepared_texts.append(text)
        all_choices.append(choices)
    # parse texts together in a batch
    parses = []
    docs = nlp.pipe(prepared_texts, batch_size=len(prepared_texts))
    for doc in docs:
        sent = list(doc.sents)[0]
        parses.append("(ROOT " + sent._.parse_string + ")")
    return parses, all_choices

class POSTree(object):
    """Penn Treebank style tree."""

    class Node(object):
        def __init__(self, tag, text):
            self.tag = tag
            self.text = text
            self.first_child = None
            self.next_sibling = None

        def __repr__(self):
            return '<%s> %s' % (self.tag, self.text)
        
        def gather_word(self):
            words = []
            if self.text:
                # the node is a word level node
                words.append(self.text)
                return words
            
            current = self.first_child
            while current != None:
                words.extend(current.gather_word())
                current = current.next_sibling
            return words
        
        def tree_to_text(self):
            words = ''
            if self.text:
                # the node is a word level node
                return "(%s %s)" % (self.tag, self.text)
            
            words += '(' + self.tag
            current = self.first_child
            while current != None:
                words += current.tree_to_text()
                current = current.next_sibling
            words += ')'
            return words
    
    def __init__(self, question:str, choices=[]):
        """Create a Penn Treebacnk style tree from plaint text.

        Using child-sibling representation.

        question: the question or parse tree.
        choices: the choices for answer
        """

        # to handle questions that have a choice, such as "on the right or on the left"
        # we do first order traverse and find (CC or), and find the node to the left and
        # right. And append the choices here
        # Each one is a list of words
        self.choices = choices

        if question.startswith("("):
            if question.startswith("(ROOT"):
                self.text = question
            else:
                self.text = "(ROOT " + question + ")"
        else:
            if question[-1] != "?":
                question += "?"
            self.text = self.get_parse_tree(question)

        self.text = self.text.replace('\n', '')
        self.text_length = len(self.text)
        self.text_pointer = 0
        self.words = []
        self.root = self.__create_tree()
        self.question = TreebankWordDetokenizer().detokenize(self.root.gather_word())
        self.question = self.question[0].upper() + self.question[1:]
    
    def get_parse_tree(self, text):
        if ' do you think' in text:
            text = text.replace(" do you think", '')
        if 'Do you think the' in text and ' or ' in text:
            text = text.replace("Do you think the", "The")
        if 'Do you' in text:
            text = text.replace("Do you", "Do I")
        if 'in this photo ' in text:
            text = text.replace(" in this photo ", ' ')
        if 'in this picture' in text:
            text = text.replace(" in this picture", '')
        if 'in this image' in text:
            text = text.replace(" in this image", '')
        if 'in this photograph' in text:
            text = text.replace(" in this photograph", '')
        if 'in the photo ' in text:
            text = text.replace(" in the photo ", ' ')
        if 'in the picture' in text:
            text = text.replace(" in the picture", '')
        if 'in the image' in text:
            text = text.replace(" in the image", '')
        if 'in the photograph' in text:
            text = text.replace(" in the photograph", '')
        if 'Of what material the' in text:
            text = text.replace("Of what material the", 'Of what material is the')
        if ',' in text:
            text, choice = text.split(",")
            text += "?"
            assert " or " in choice
            if "?" in choice:
                choice = choice.replace("?", '')
            self.choices = [c.strip().split() for c in choice.strip().split(" or ")]
        if text.startswith("In front of"):
            text = text.replace("In front of ", '')
            text = text[:-1] + " in front of?"
        if 'in front or behind' in text:
            # the parser can not correctly parse in front or behind
            text = text.replace("in front or behind", "a or b")
        if 'behind or in front of' in text:
            text = text.replace("behind or in front of", 'a or b')
        doc = nlp(text)
        sent = list(doc.sents)[0]
        return "(ROOT " + sent._.parse_string + ")"

    def __create_tree(self):
        parent = None
        token = self.__next_token()
        if token == "(":
            # start a new tree
            tag = self.__next_token()
            if tag in WORDLEVEL_TAGS or tag.startswith("PRP") or \
                tag.startswith("WP"):
                # for word level tags, we also need the text
                text = self.__next_token().lower()
                parent = self.Node(tag, text)
                self.words.append(text)
                token = self.__next_token()
                assert token == ")"
            else:
                # for phrase level tags, the text are None
                parent = self.Node(tag, None)
                parent.first_child = self.__create_tree()
                child = parent.first_child
                if child != None:
                    while True:
                        child.next_sibling = self.__create_tree()
                        child = child.next_sibling
                        if child == None:
                            break
        
        return parent
    
    def __next_token(self):
        # get a token in the input text, not including the spacing
        end = self.text_pointer
        while end < self.text_length and self.text[end] == ' ':
            end += 1
        
        if end == self.text_length:
            return None

        if self.text[end] in ('(', ')'):
            token = self.text[end]
            end += 1
        else:
            start = end
            end += 1
            while end < self.text_length and self.text[end] not in ('(', ')', ' '):
                end += 1
            token = self.text[start:end]
        self.text_pointer = end
        return token
        
    def __replace_qmark_with_period(self):
        # replace the last question mark into period
        child = self.root.first_child.first_child
        assert(child.tag != '.')
        while child.next_sibling != None and child.next_sibling.tag != '.':
            child = child.next_sibling
        if child.next_sibling == None:
            # the question mark might be hidden in the last child
            subchild = child.first_child
            assert(subchild.tag != '.')
            while subchild.next_sibling != None \
                    and subchild.next_sibling.tag != '.':
                subchild = subchild.next_sibling
            if subchild.next_sibling == None:
                raise RuntimeError("Please include a question mark at the end")
            else:
                period = self.__delete_tree(subchild, subchild.next_sibling)
                period.text = '.'
                self.__insert_after(period, child)
        else:
            child.next_sibling.text = '.'
    
    def __check_VB(self, node):
        # check if node is a VB
        if node.tag in VB_TAGS:
            return True
        if node.text == None:
            return False
        if node.text in VB_WORDS:
            node.tag = 'VB'
            return True
        return False
    
    def __check_ADVP(self, prenode, node):
        # (ADVP (JJ next) (PP (IN to) (NP (DT the) (NN countertop))
        while node != None and node.tag == 'ADVP':
            prenode = node
            node = node.next_sibling
        return prenode, node
    
    def __create_answer_node(self, before_text='', after_text=''):
        # create an answer node with text before and after it
        answer_text = ' '.join([before_text, '**blank**', after_text]).strip()
        node = self.Node('ANS', answer_text)
        return node
    
    def __insert_after(self, srcnode, dstnode):
        assert srcnode != None and dstnode != None
        srcnode.next_sibling = dstnode.next_sibling
        dstnode.next_sibling = srcnode
        return srcnode
    
    def __insert_as_first_child(self, srcnode, dstnode):
        assert srcnode != None and dstnode != None
        srcnode.next_sibling = dstnode.first_child
        dstnode.first_child = srcnode
        return srcnode
    
    def __delete_tree(self, prenode, node):
        if node == None:
            return node
        if prenode.first_child == node:
            prenode.first_child = node.next_sibling
        else:
            prenode.next_sibling = node.next_sibling
        node.next_sibling = None
        return node
    
    def __find_choices(self, node):
        if node == None:
            return []
        if 'either' in node.gather_word():
            return []
        if 'or' in node.gather_word() and \
            (self.question.startswith("Is there") or
            self.question.startswith("Are there")):
            return []
        if 'or' in node.gather_word() and \
            self.question.startswith("Do i see"):
            return []
        # find (CC, or)
        has_cc_or = False
        left = []
        right = []
        cur = node.first_child
        while cur != None:
            if cur.tag == "CC" and cur.text == "or":
                has_cc_or = True
                cur = cur.next_sibling
                continue
            if cur.tag == "DT" and cur.text == "any":
                cur = cur.next_sibling
                continue
            if cur.tag == "RB" and cur.text == "maybe":
                cur = cur.next_sibling
                continue
            if has_cc_or:
                right.append(cur)
            else:
                left.append(cur)
            cur = cur.next_sibling
        if has_cc_or:
            left_tags = [subnode.tag for subnode in left if subnode.tag != 'DT']
            right_tags = [subnode.tag for subnode in right if subnode.tag != 'DT']
            if left_tags == ["JJ", "CC", "JJ"] or \
                left_tags == ["JJ", "NN"] or \
                left_tags == ["JJ", "NNS"] or \
                left_tags == ["JJ", "JJ"] or \
                left_tags == ['NN', 'NNS'] or \
                left_tags == ['VBN', 'NN'] or \
                left_tags == ['RB', 'JJ'] or \
                left_tags == ['JJ', 'VBN'] or \
                left_tags == ['JJ', 'NN', 'NNS']:
                # (JJ black) (CC and) (JJ white) (CC or) (JJ colorful)
                # (JJ blue)(NN briefcase)(CC or)(NN backpack)
                # (JJ black)(NNS ties)(CC or)(NNS glasses)
                # (JJ short)(JJ sleeved)(CC or)(JJ sleeveless)
                # (NN bar)(NNS stools)(CC or)(NNS tables)
                # (VBN closed) (NN window) (CC or) (NN door)
                # (RB long)(JJ sleeved)(CC or)(JJ sleeveless)
                # (JJ short)(VBN sleeved)(CC or)(JJ sleeveless)
                # (JJ white)(NN soccer)(NNS balls)(CC or)(NNS frisbees)
                left_choices = []
                for subnode in left:
                    left_choices += subnode.gather_word()
            elif len(left) == 1:
                left_choices = left[0].gather_word()
            else:
                raise ValueError("Unknown left choice %s" % left)
            if right_tags == ["JJ", "CC", "JJ"] or \
                right_tags == ["JJ", "NN"] or \
                right_tags == ["JJ", "NNS"] or \
                right_tags == ['JJ', 'JJ'] or \
                right_tags == ['NN', 'NNS'] or \
                right_tags == ['VBN', 'NN'] or \
                right_tags == ['RB', 'JJ'] or \
                right_tags == ['JJ', 'VBN'] or \
                right_tags == ['JJ', 'NN', 'NNS']:
                right_choices = []
                for subnode in right:
                    right_choices += subnode.gather_word()
            elif len(right) == 1:
                right_choices = right[0].gather_word()
            else:
                raise ValueError("Unknown right choice %s" % right)
            return [left_choices, right_choices]
        else:
            return []
    
    def __convert_WH_to_answer(self, WH):
        words = WH.gather_word()
        WH_text = ' '.join(words)
        if WH_text == 'how old':
            WH.first_child = self.__create_answer_node(after_text='years old')
        elif WH_text == 'how long':
            WH.first_child = self.__create_answer_node(after_text='in length')
        elif WH_text == 'how clean':
            WH.first_child = self.__create_answer_node(after_text='in cleanliness')
        elif WH_text == 'how tall':
            WH.first_child = self.__create_answer_node(after_text='in height')
        elif WH_text == 'how heavy':
            WH.first_child = self.__create_answer_node(after_text='in weight')
        elif WH_text == 'how hard':
            WH.first_child = self.__create_answer_node(after_text='in hardness')
        elif WH_text == 'how wide':
            WH.first_child = self.__create_answer_node(after_text='in width')
        elif WH_text in ['how large', 'how big']:
            WH.first_child = self.__create_answer_node(after_text='in size')
        elif WH_text == 'how deep':
            WH.first_child = self.__create_answer_node(after_text='in depth')
        elif WH_text == 'how real':
            WH.first_child = self.__create_answer_node(after_text='in trueness')
        elif WH_text == 'how fat':
            WH.first_child = self.__create_answer_node(after_text='in body size')
        elif WH_text == 'how thick':
            WH.first_child = self.__create_answer_node(after_text='in thickness')
        elif WH_text == 'who':
            WH.first_child = self.__create_answer_node(before_text='the', after_text='is the person who')
        elif WH_text == 'where':
            # (ROOT (SBARQ (WHADVP (WRB Where)) (SQ (VBZ is) (NP (DT the) (NN cat))) (. ?)))
            WH.first_child = self.__create_answer_node(before_text='the location')
        elif WH.tag in ('WHADJP', 'WHADVP'):
            # (ROOT (SBARQ (WHADVP (WRB How)) (SQ (VBZ is) (NP (NP (DT the) (NN vehicle)) (PP (IN to) (NP (NP (DT the) (NN right)) (PP (IN of) (NP (NP (DT the) (NN vehicle)) (PP (IN below) (NP (DT the) (NN flag)))))))) (VP (VBN called))) (. ?)))
            WH.first_child = self.__create_answer_node()
        elif WH.tag == 'WHNP':
            # (WHNP (WP What))
            # (WHNP (WHNP (WP What) (NN kind)) -> The kind
            # (WHNP (WDT Which) (NN color))
            # (WHNP (WP What) (NN vehicle))
            if 'the gender of' in self.question:
                WH.first_child = self.__create_answer_node()
            elif self.question.startswith("What is the height") or \
                self.question.startswith("What is the size") or \
                self.question.startswith("What height") or \
                self.question.startswith("What size") or \
                self.question.startswith("What is the name of") or \
                self.question.startswith("What is the width"):
                WH.first_child = self.__create_answer_node()
            else:
                answer_prefix = ' '.join(['the'] + words[1:])
                WH.first_child = self.__create_answer_node(before_text=answer_prefix)
        elif WH.tag == 'WHPP':
            whfcns_words = WH.first_child.next_sibling.gather_word()
            whfcns_text = ' '.join(whfcns_words)
            if whfcns_text == 'where':
                # (ROOT (SBARQ (WHPP (IN From) (WHADVP (WRB where))) (SQ (VBZ does) (NP (DT the) (NN steam)) (VP (VB come))) (. ?)))
                answer_prefix = 'the location'
            elif whfcns_text in ['who', 'whom']:
                answer_prefix = 'the person'
            elif WH.first_child.next_sibling.tag == 'WHNP':
                # WHPP: (WHPP (IN Of) (WHNP (WP what) (NN color))) ...
                answer_prefix = ' '.join(['the'] + whfcns_words[1:])
            else:
                raise ValueError('Unknown WH structure! %s' % WH.tree_to_text())
            WH.first_child.next_sibling.first_child = self.__create_answer_node(before_text=answer_prefix)
        else:
            raise ValueError('Unknown WH structure! %s' % WH)
        return WH
        
    def __adjust_SQ_question(self, SQ):
        VB = SQ.first_child
        assert self.__check_VB(VB)
        auxiliary = VB.text
        # keep the auxiliary
        answer = self.__create_answer_node(before_text=auxiliary)

        # move answer after first NP
        NP = VB.next_sibling
        while NP.tag != 'NP':
            NP = NP.next_sibling
        # there still might be some bugs
        # check if this position is valid
        # if either of these following
        # situation occurs then it is invalid
        VALID = True
        if NP.next_sibling.tag == "SBAR":
            VALID = False
        # if invalid, then find the first NP sibling of NP
        if not VALID:
            NP = NP.first_child
            while NP != None and NP.tag != 'NP':
                NP = NP.next_sibling
        if NP == None:
            # (ROOT (SQ (VBZ Is) (NP (DT the) (NN device)) (SBAR (WHNP (WDT that)) (S (VP (VBZ looks) (ADJP (JJ gray)) (ADVP (IN off) (CC or) (IN on))))) (. ?)))
            # (ROOT (SQ (VBZ Is) (NP (DT the) (NN shirt)) (SBAR (WHNP (WDT that)) (S (VP (VBZ looks) (ADJP (JJ soft) (JJ small) (CC and) (JJ black))))) (. ?)))
            NP = VB.next_sibling
            if NP.next_sibling.tag == "SBAR" and \
                (NP.next_sibling.next_sibling == None or \
                NP.next_sibling.next_sibling.tag == '.'):
                S = NP.next_sibling.first_child.next_sibling
                assert S.tag == "S" and S.first_child.tag == 'VP'
                NP = S.first_child.first_child
                while NP.next_sibling != None and NP.tag not in ['ADJP', 'VP']:
                # while NP.next_sibling != None:
                    NP = NP.next_sibling
                if NP == None:
                    raise ValueError("Unknown SQ structure ")
                elif NP.tag == 'ADJP':
                    if NP.first_child.next_sibling != None:
                        NP = NP.first_child
                elif NP.tag == 'VP':
                    # (ROOT (SQ (VBZ Is) (NP (DT the) (NN man)) (SBAR (WHNP (WDT that)) (S (VP (VBZ is) (VP (VBG sitting) (ADJP (DT both) (JJ Caucasian) (CC and) (JJ old)))))) (. ?)))
                    preADJP = NP.first_child
                    while preADJP.next_sibling != None and preADJP.next_sibling.tag != 'ADJP':
                        preADJP = preADJP.next_sibling
                    if preADJP.next_sibling == None:
                        raise ValueError("Unknown SQ structure ")
                    else:
                        NP = preADJP
                elif NP.tag == 'PP':
                    # (ROOT(SQ(VBZ is)(NP(DT the)(NN sink))(SBAR(WHNP(WDT that))(S(VP(VBZ is)(RB not)(PP(IN on)(ADJP(JJ curved)(CC and)(JJ black))))))(. .)))
                    # (ROOT(SQ(VBZ is)(NP(DT the)(NN doughnut))(SBAR(WHNP(WDT which))(S(VP(VBZ is)(PP(IN to)(NP(NP(DT the)(NN left))(PP(IN of)(NP(DT the)(NN person)))))(ADJP(JJ large)(CC and)(JJ round)))))(. .)))
                    preADJP = NP.first_child
                    while preADJP.next_sibling != None and \
                        preADJP.next_sibling.tag != 'ADJP':
                        preADJP = preADJP.next_sibling
                    assert preADJP.next_sibling.tag == 'ADJP'
                    NP = preADJP
                else:
                    raise ValueError("Unknown SQ structure ")
            elif NP.next_sibling.tag == "SBAR" and \
                NP.next_sibling.next_sibling.tag == 'VP':
                NP = NP.next_sibling
            else:
                raise ValueError("Unknown SQ structure ")
        # (ROOT (SQ (VBP Are) (DT both) (NP (NP (DT the) (NN appliance)) (SBAR (WHNP (WDT that)) (S (VP (VBZ looks) (ADJP (JJ rectangular)))))) (CC and) (NP (NP (NP (DT the) (NN appliance)) (PP (IN to) (NP (NP (DT the) (NN right)) (PP (IN of) (NP (DT the) (NN stove)))))) (VP (VBN made) (PP (IN of) (NP (JJ stainless) (NN steel))))) (. ?)))
        # In some situation there are still CC and NP behind first NP
        if NP.next_sibling != None and \
            NP.next_sibling.tag == "CC" and \
            NP.next_sibling.next_sibling != None and \
            NP.next_sibling.next_sibling.tag == "NP":
            NP = NP.next_sibling.next_sibling
            if NP.first_child.tag == "NP" and \
                NP.first_child.next_sibling != None:
                NP = NP.first_child
        # (ROOT (SQ (VBP Are) (DT both) (NP (NP (DT the) (JJ white) (NN thing)) (PP (IN to) (NP (NP (DT the) (NN right)) (PP (IN of) (NP (DT the) (NN chair)))))) (CC and) (VP (NP (NP (DT the) (NN toilet) (NN paper)) (PP (IN to) (NP (NP (DT the) (NN right)) (PP (IN of) (NP (DT the) (NN toilet)))))) (VP (VBN made) (PP (IN of) (NP (NN paper))))) (. ?)))
        if NP.next_sibling != None and \
            NP.next_sibling.tag == "CC" and \
            NP.next_sibling.next_sibling != None and \
            NP.next_sibling.next_sibling.tag == "VP" and \
            NP.next_sibling.next_sibling.first_child != None and \
            NP.next_sibling.next_sibling.first_child.tag == "NP":
            NP = NP.next_sibling.next_sibling.first_child
        answer = self.__insert_after(answer, NP)
        if not self.choices:
            # check if there are choices
            self.choices = self.__find_choices(answer.next_sibling)
            if self.choices:
                self.__delete_tree(answer, answer.next_sibling)
            elif answer.next_sibling != None:
                # also check the first child of next sibling
                self.choices = self.__find_choices(answer.next_sibling.first_child)
                if self.choices:
                    self.__delete_tree(answer.next_sibling, answer.next_sibling.first_child)
                elif answer.next_sibling.first_child != None:
                    # also check the second child of next sibling
                    # (ROOT (SQ (MD Could) (NP (DT this) (NN place)) (VP (VB be) (NP (NP (DT a) (NN park)) (CC or) (NP (DT a) (NN beach)))) (. ?)))
                    self.choices = self.__find_choices(answer.next_sibling.first_child.next_sibling)
                    if self.choices:
                        if self.__check_VB(answer.next_sibling.first_child):
                            subvb = answer.next_sibling.first_child.text
                            ans_words = answer.text.split(" ")
                            answer.text = ' '.join(ans_words[:1] + [subvb] + ans_words[1:])
                            self.__delete_tree(answer, answer.next_sibling)
                        elif answer.next_sibling.first_child.tag == 'DT' and \
                            answer.next_sibling.first_child.text == 'a':
                            # (ROOT(SQ(VBZ is)(NP(EX there))(ANS is **blank**)(NP(DT a)(NML(NML(JJ black)(NN keyboard))(CC or)(NML(JJ remote)(NN control))))(. .)))
                            self.__delete_tree(answer, answer.next_sibling)
                    elif answer.next_sibling.first_child.next_sibling != None and \
                        answer.next_sibling.first_child.next_sibling.first_child != None and \
                        self.__check_VB(answer.next_sibling.first_child):
                        # (ROOT (SQ (VBZ Is) (NP (DT the) (NN fence)) (VP (VBN made) (PP (IN of) (NP (NN cement) (CC or) (NN aluminum)))) (. ?)))
                        self.choices = self.__find_choices(answer.next_sibling.first_child.next_sibling.first_child.next_sibling)
                        if self.choices:
                            self.__delete_tree(answer.next_sibling.first_child.next_sibling.first_child, \
                                answer.next_sibling.first_child.next_sibling.first_child.next_sibling)
                        elif answer.next_sibling.tree_to_text().startswith("(VP(VB seem)(S(VP(TO to)(VP(VB be)"):
                            # (ROOT (SQ (VBZ Does) (NP (NP (DT the) (NN player)) (PP (IN on) (NP (NP (DT the) (NN side)) (PP (IN of) (NP (DT the) (NN home) (NN plate)))))) (VP (VB seem) (S (VP (TO to) (VP (VB be) (VP (VBG playing) (CC or) (VBG waiting)))))) (. ?)))
                            self.choices = self.__find_choices(answer.next_sibling.first_child.next_sibling.first_child.first_child.next_sibling.first_child.next_sibling)
                            if self.choices:
                                self.__delete_tree(answer.next_sibling.first_child.next_sibling.first_child.first_child.next_sibling.first_child, \
                                    answer.next_sibling.first_child.next_sibling.first_child.first_child.next_sibling.first_child.next_sibling)
                elif answer.next_sibling.tag in ['NN', 'NNS', 'JJ'] and \
                    answer.next_sibling.next_sibling != None and \
                    answer.next_sibling.next_sibling.tag == 'CC' and \
                    answer.next_sibling.next_sibling.text == 'or' and \
                    answer.next_sibling.next_sibling.next_sibling != None and \
                    answer.next_sibling.next_sibling.next_sibling.tag in ['NN', 'NNS', 'JJ']:
                    # (ROOT (SQ (VBP Are) (NP (DT the) (NNS letters)) (SBAR (WHNP (WDT that)) (S (VP (VBP are) (RB not) (ADJP (JJ small) (NN orange) (CC or) (JJ white))))) (. ?)))
                    self.choices = [[answer.next_sibling.text], [answer.next_sibling.next_sibling.next_sibling.text]]
                    self.__delete_tree(answer, answer.next_sibling)
                    self.__delete_tree(answer, answer.next_sibling)
                    self.__delete_tree(answer, answer.next_sibling)
        if not self.choices:
            # SQ questions always have choices
            self.choices = [['yes'], ['no']]
        if len(self.choices[1]) > 4 and \
            (self.choices[1][:4] == ['to', 'the', 'left', 'of'] or
            self.choices[1][:4] == ['to', 'the', 'right', 'of'] or
            self.choices[1][:4] == ['on', 'the', 'left', 'of'] or
            self.choices[1][:4] == ['on', 'the', 'right', 'of']):
            left = self.choices[1][4:]
            left = ' '.join(left)
            answer.text += ' ' + left
            self.choices[1] = self.choices[1][:4]
        if self.choices == [['a'], ['b']]:
            self.choices = [["in", "front", "of"], ["behind"]]
        self.__delete_tree(SQ, VB)
        return SQ
    
    def __adjust_SQ_in_SBARQ(self, SQ, WH):
        prefirst, first = self.__check_ADVP(SQ, SQ.first_child)

        # SQ = VP
        # (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ holds) (NP (DT the) (NN umbrella)))) (. ?))
        if first.tag == 'VP':
            if self.__check_VB(first.first_child) and first.first_child.next_sibling == None:
                # unless there is only one VB in VP
                # (SQ (VP (VBZ is)) ...
                # (ROOT (SBARQ (WHADJP (WRB How) (JJ old)) (SQ (VP (VBZ is)) (NP (NP (DT the) (NN girl)) (SBAR (WHNP (WDT that)) (S (VP (VBZ is) (VP (VBG wearing) (NP (DT a) (NN sweatshirt)))))))) (. ?)))
                VB = first.first_child
                self.__delete_tree(prefirst, first)
                self.__insert_as_first_child(VB, prefirst)
                first = VB
            elif self.__check_VB(first.first_child) and \
                first.first_child.text in VB_WORDS and \
                first.first_child.next_sibling.tag == 'NP' and \
                first.first_child.next_sibling.next_sibling == None:
                # (ROOT (SBARQ (ADVP (IN Inside)) (WHNP (WP what)) (SQ (VP (VBZ is) (NP (DT the) (NN pizza)))) (. ?)))
                NP = self.__delete_tree(first.first_child, first.first_child.next_sibling)
                VB = self.__delete_tree(first, first.first_child)
                self.__delete_tree(prefirst, first)
                VB = self.__insert_as_first_child(VB, prefirst)
                self.__insert_after(NP, VB)
                first = VB
            else:
                return SQ
        
        # SQ = NP + VP
        if (first.tag == 'NP' and first.next_sibling != None 
                and first.next_sibling.tag == 'VP' and first.next_sibling.next_sibling == None):
            return SQ
        
        if not self.__check_VB(first):
            raise ValueError('First child of SQ in SBARQ is not VB*/MD, WH: %s' % WH.tree_to_text())
        
        # process 's 're 've
        if first.text == "'s":
            first.text = 'is'
        elif first.text == "'re":
            first.text = 'are'
        elif first.text == "'ve":
            first.text = 'have'
        
        presecond, second = self.__check_ADVP(first, first.next_sibling)

        # SQ = VB* + [ADVP]
        # (SBARQ (WHNP (WHADJP (WRB How) (JJ many)) (NNS horses)) (SQ (VBP are) (ADVP (RB there))) (. ?))
        if second == None:
            return SQ
        
        # process RB(not) and auxiliary do/does/did
        if second.tag == 'RB' and second.text in ["n't", "not"]:
            # (SBARQ (WHNP (WP Who)) (SQ (VBP do) (RB n't) (VP (VBG hold) (NP (NN umbrella)))) (. ?))
            if first.text == 'ca':
                first.text = 'can not'
            else:
                first.text += ' not'
            self.__delete_tree(presecond, second)
            presecond, second = self.__check_ADVP(first, first.next_sibling)
        else:
            if first.text in ['do', 'does', 'did']:
                # who eat apple?
                first.text = ''
        
        # SQ = VB*+PP/ADJP/VP
        # (SBARQ (WHNP (WP What) (NN vehicle)) (SQ (VBP is) (PP (IN behind) (PP (IN of) (NP (DT the) (NNS trees))))) (. ?))
        # --> The vehicle truck is behind of the trees.
        # (SBARQ (WHNP (WHNP (WP What) (NN kind)) (PP (IN of) (NP (NN appliance)))) (SQ (VBZ is) (PP (IN under) (NP (DT the) (NNS cabinets)))) (. ?))
        # --> The kind of appliance countertop is under the cabinets.
        # (ROOT (SBARQ (WHNP (WP What)) (SQ (VBZ is) (PP (IN in) (NP (NP (NN front)) (PP (IN of) (NP (DT the) (NN wall))))) (SBAR (WHNP (WDT that)) (S (VP (VBZ is) (RB not) (ADJP (JJ short)))))) (. ?)))
        # --> The couch is in front of the wall that is not short.
        if second.next_sibling == None and second.tag in ['PP', 'ADJP', 'VP']:
            return SQ
        # (ROOT (SBARQ (WHNP (WP What)) (SQ (VBZ is) (S (VP (VBG throwing) (NP (DT the) (NN baseball))))) (. ?)))
        if second.next_sibling == None and second.tag == 'S' and \
            second.first_child.tag == 'VP' and second.first_child.next_sibling == None:
            S = self.__delete_tree(first, second)
            self.__insert_after(S.first_child, first)
            return SQ
        
        if second.next_sibling != None and \
            second.next_sibling.tag in ["SBAR", "VP", "PP"] and \
            second.tag in ['PP', 'ADJP', 'VP']:
            # (ROOT (SBARQ (WHNP (WP What)) (SQ (VBZ is) (PP (IN on) (NP (DT the) (NN side) (NN walk))) (VP (VBN made) (PP (IN of) (NP (NN concrete))))) (. ?)))
            # (ROOT(SBARQ(SQ(VBZ is)(PP(IN on)(NP(DT the)(NN stove)))(PP(IN in)(NP(DT the)(NN center))))(. .)))
            return SQ
        
        # SQ = VB* + NP
        #      |     |
        #     first second
        # (SBARQ (WHNP (WP What) (NN vehicle)) (SQ (VBP are) (NP (DT the) (NNS trees)) (PP (IN behind) (PP (IN of)))) (. ?))
        # --> The trees are behind of the vehicle truck.
        if second.next_sibling == None and second.tag == 'NP':
            fc = second.first_child

            # second = NP + ?
            #          |    |
            #          fc   sc
            if (fc.tag == 'NP' and fc.next_sibling != None
                    and fc.next_sibling.next_sibling == None):
                sc = fc.next_sibling
                # (ROOT (SBARQ (WHNP (WP What)) (SQ (VBZ is) (NP (NP (NP (DT the) (NN faucet)) (PP (IN in) (NP (NN front)))) (PP (IN of)))) (. ?)))
                if fc.first_child.tag == "NP" and \
                    fc.first_child.next_sibling.tag == "PP" and \
                    fc.first_child.next_sibling.gather_word() == ['in', 'front'] and \
                    sc.first_child.text == 'of':
                    VB = self.__delete_tree(prefirst, first)
                    self.__insert_after(VB, fc.first_child)
                    return SQ
                if ((sc.tag == 'PP' and WH.tag == 'WHADVP')
                        or (sc.tag == 'PP' and sc.first_child.tag == 'IN'
                            and sc.first_child.next_sibling == None)
                        or (sc.tag == 'NP' and ' '.join(fc.gather_word()) == 'there')
                        or (sc.tag == 'ADJP')
                        or (sc.tag == 'SBAR' and sc.first_child.tag == 'WHADVP')):
                    self.__delete_tree(presecond, second)
                    VB = self.__delete_tree(prefirst, first)
                    self.__insert_after(VB, fc)
                    return SQ
                # find the last (VP
                elif sc.tag == 'VP' and sc.first_child.tag in ['VBN', 'VBG']:
                    # (ROOT (SBARQ (WHNP (WP What)) (SQ (VBZ is) (NP (NP (NP (DT the) (NN appliance)) (SBAR (WHNP (WDT that)) (S (VP (VBZ is) (PP (IN beneath) (NP (NP (DT the) (NN appliance)) (PP (IN below) (NP (DT the) (NNS cabinets))))))))) (VP (VBN called)))) (. ?)))
                    VB = self.__delete_tree(prefirst, first)
                    self.__insert_after(VB, fc)
                    return SQ
                elif sc.tag == 'SBAR' and \
                    sc.first_child.next_sibling != None and \
                    sc.first_child.next_sibling.tag == 'S' and \
                    sc.first_child.next_sibling.first_child.tag == 'VP':
                    # (ROOT (SBARQ (WHNP (WP What)) (SQ (VBZ is) (NP (NP (DT the) (NN man)) (SBAR (WHNP (WDT that)) (S (VP (VBZ is) (PP (IN to) (NP (DT the) (NN left))) (PP (IN of) (NP (NP (DT the) (NN boy)) (VP (VBG leaning) (PP (RP on)))))))))) (. ?)))
                    # (ROOT (SBARQ (WHNP (WP What)) (SQ (VBZ is) (NP (NP (DT the) (NN device)) (SBAR (WHNP (WDT that)) (S (VP (VBZ is) (RB not) (PP (IN on))))))) (. ?)))
                    last = sc.first_child.next_sibling.first_child.first_child
                    while last.next_sibling != None:
                        last = last.next_sibling
                    if last.tag == 'PP' and \
                        last.first_child.next_sibling != None:
                        NP = last.first_child.next_sibling
                        if NP.tag == 'NP' and \
                            NP.first_child != None and \
                            NP.first_child.tag == 'NP' and \
                            NP.first_child.next_sibling != None and \
                            NP.first_child.next_sibling.tag == 'VP':
                            VB = self.__delete_tree(prefirst, first)
                            self.__insert_after(VB, NP.first_child)
                            return SQ
                    elif last.tag == 'PP' and \
                        last.first_child.next_sibling == None:
                        # (ROOT (SBARQ (WHNP (WP What)) (SQ (VBZ is) (NP (NP (DT the) (NN device)) (SBAR (WHNP (WDT that)) (S (VP (VBZ is) (RB not) (PP (IN on))))))) (. ?)))
                        VB = self.__delete_tree(prefirst, first)
                        self.__insert_after(VB, last)
                        return SQ
                    else:
                        ValueError('Unknown SQ structure in SBARQ!')
            VB = self.__delete_tree(prefirst, first)
            self.__insert_after(VB, second)
            return SQ
        
        # SQ = VB* + NP + ? 
        #      |     |    |
        #    first second third
        # (SBARQ (WHNP (WP What)) (SQ (VBZ 's) (NP (DT the) (NN wallet)) (VP (VBN made) (PP (IN of)))) (. ?))
        # --> The wallet is made of the leather.
        if second.tag == 'NP' and second.next_sibling != None:
            prethird, third = self.__check_ADVP(second, second.next_sibling)
            # SQ = VB* + NP + ADVP
            if third == None:
                VB = self.__delete_tree(prefirst, first)
                self.__insert_after(VB, second)
                return SQ
            
            if third.next_sibling == None:
                if ((third.tag in ['ADJP', 'PP', 'NP', 'VP'])
                        or (third.tag == 'S'
                            and third.tree_to_text().startswith('(S(VP(TO to)(VP(VB')
                            )):
                    VB = self.__delete_tree(prefirst, first)
                    self.__insert_after(VB, second)
                    return SQ
            
            # (ROOT (SBARQ (WHADVP (WRB How)) (SQ (VBZ is) (NP (NP (DT the) (NN piece)) (PP (IN of) (NP (NN furniture))) (PP (IN below) (NP (DT the) (NN mirror)))) (SBAR (S (NP (DT the) (NN television)) (VP (VBZ is) (PP (IN to) (NP (NP (DT the) (NN right)) (PP (IN of)))) (VP (VBN called)))))) (. ?)))
            # (ROOT (SBARQ (WHNP (WP What) (NN color)) (SQ (VBZ is) (NP (DT the) (NN serving) (NN tray)) (SBAR (WHNP (WDT that)) (S (VP (VBZ looks) (ADJP (JJ rectangular)))))) (. ?)))
            if third.next_sibling == None and \
                third.tag == 'SBAR' and \
                third.first_child.tag == "S" and \
                third.first_child.first_child.tag == "NP" and \
                third.first_child.first_child.next_sibling.tag == "VP":
                VP = third.first_child.first_child.next_sibling
                assert self.__check_VB(VP.first_child) and VP.first_child.next_sibling != None
                last = VP.first_child.next_sibling
                while last.next_sibling.next_sibling != None:
                    last = last.next_sibling
                VB = self.__delete_tree(prefirst, first)
                self.__insert_after(VB, last)
                return SQ
            if third.next_sibling == None and \
                third.tag == 'SBAR' and \
                third.first_child.next_sibling != None and \
                third.first_child.next_sibling.tag == 'S' and \
                third.first_child.next_sibling.first_child.tag == 'VP':
                VB = self.__delete_tree(prefirst, first)
                self.__insert_after(VB, third.first_child.next_sibling)
                return SQ
        raise ValueError('Unknown SQ structure in SBARQ!')
    
    def __insert_WH_into_SQ(self, WH, SQ):
        prefirst, first = self.__check_ADVP(SQ, SQ.first_child)

        if first.next_sibling == None:
            # SQ = VP
            # (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ holds) (NP (DT the) (NN umbrella)))) (. ?))
            # --> The girl holds the umbrella.
            if first.tag == 'VP':
                self.__insert_as_first_child(WH, SQ)
                return SQ
            
            # SQ = NP
            # (SBARQ (WHNP (WP What) (NN vehicle)) (SQ (VBP are) (NP (DT the) (NNS trees)) (PP (IN behind) (PP (IN of)))) (. ?))
            # --> The trees are behind of the vehicle truck.
            if first.tag == 'NP':
                self.__insert_after(WH, first)
                return SQ
            
            # SQ = VB*
            # (SBARQ (WHNP (WP Who)) (SQ (VP (VBZ holds) (NP (DT the) (NN umbrella)))) (. ?))
            # --> The girl holds the umbrella.
            if self.__check_VB(first):
                self.__insert_as_first_child(WH, SQ)
                return SQ
            
            raise ValueError('Unknown SQ structure!')

        presecond, second = self.__check_ADVP(first, first.next_sibling)

        # SQ = VB* + ADVP
        if self.__check_VB(first) and second == None:
            self.__insert_as_first_child(WH, SQ)
            return SQ
        
        # SQ = VB* + VP/PP/ADJP
        #      |     |
        #    first  second
        if (self.__check_VB(first) and second.next_sibling == None 
                and second.tag in ('VP', 'PP', 'ADJP')):
            self.__insert_as_first_child(WH, SQ)
            return SQ
        
        if (self.__check_VB(first) and second.next_sibling != None and second.next_sibling.tag in ["SBAR", "VP", "PP"] 
                and second.tag in ('VP', 'PP', 'ADJP')):
            # (ROOT (SBARQ (WHNP (WP What)) (SQ (VBZ is) (PP (IN in) (NP (NP (NN front)) (PP (IN of) (NP (DT the) (NN wall))))) (SBAR (WHNP (WDT that)) (S (VP (VBZ is) (RB not) (ADJP (JJ short)))))) (. ?)))
            self.__insert_as_first_child(WH, SQ)
            return SQ
        
        # if until now WH still involves person, we need to change the answer
        if WH.first_child.text == 'the **blank** is the person who':
            WH.first_child.text = 'the person **blank**'
        
        prethird, third = self.__check_ADVP(second, second.next_sibling)

        # SQ = NP + VB* + [ADVP]
        #      |    |      
        #    first second 
        if (first.tag == 'NP' and self.__check_VB(second) and 
                (second.next_sibling == None or third == None)):
            if self.words[0] == 'where':
                second = self.__insert_after(self.Node('IN', 'in'), second)
            self.__insert_after(WH, second)
            return SQ
        
        # SQ = NP + VP
        #      |    |
        #    first second
        if (first.tag == 'NP' and second.tag == 'VP' 
                and second.next_sibling == None):
            if WH.tag in ('WHNP', 'WHADJP'):
                self.__insert_as_first_child(WH, SQ)
                return SQ
            if WH.tag in ['WHPP', 'WHADVP']:
                self.__insert_after(WH, second)
                return SQ
        
        # SQ = NP + SBAR
        #      |    |
        #    first second
        # (ROOT (SBARQ (WHADVP (WRB How)) (SQ (VBZ is) (NP (NP (DT the) (NN piece)) (PP (IN of) (NP (NN furniture))) (PP (IN below) (NP (DT the) (NN mirror)))) (SBAR (S (NP (DT the) (NN television)) (VP (VBZ is) (PP (IN to) (NP (NP (DT the) (NN right)) (PP (IN of)))) (VP (VBN called)))))) (. ?)))
        if first.tag == "NP" and second.tag == "SBAR":
            self.__insert_after(WH, second)
            return SQ
        
        if third == None:
            print(first.tree_to_text())
            print(second.tree_to_text())
            print(second.next_sibling)
            raise ValueError('Unknown SQ structure!')
        
        # SQ = NP + VB* + ?
        #      |    |     |
        #   first second third
        if first.tag == 'NP' and self.__check_VB(second) and third.next_sibling == None:

            # SQ = NP + VB* + VP
            # (SBARQ (WHADVP (WRB Where)) (SQ (VBZ does) (NP (DT the) (NN man)) (VP (VB stand) (PP (IN on)))) (. ?))
            # (ROOT (SBARQ (WHNP (WP Who)) (SQ (VBZ is) (NP (DT the) (NN baby)) (VP (VBG sitting) (ADVP (RB atop)))) (. ?)))
            if third.tag == 'VP':
                VB = second
                VP = third
                while (self.__check_VB(VP.first_child) and VP.first_child.next_sibling != None
                        and VP.first_child.next_sibling.tag == 'VP'):
                    VB = VP.first_child
                    VP = VB.next_sibling
                # VP = VBN + [...]
                #      |
                #      fc
                _, fc = self.__check_ADVP(VP, VP.first_child)
                if ((VB.text != '' 
                        and VB.text.split()[0] in ('is', 'are', 'was', 'were'))
                        and fc.tag == 'VBN'):
                    if WH.tag == 'WHADVP' and self.words[0] == 'how':
                        # WH = self.__prefix_by_to_WH(WH)
                        self.__insert_after(WH, VP)
                        return SQ
                    if WH.tag == 'WHADVP' and self.words[0] in ('why', 'where'):
                        self.__insert_after(WH, VP)
                        return SQ
                # VP = VB*
                #      |
                #      fc
                if self.__check_VB(fc) and fc.next_sibling == None:
                    self.__insert_after(WH, VP)
                    return SQ
                # VP = VB* + ?
                #      |     |
                #      fc    sc
                if (self.__check_VB(fc) and fc.next_sibling != None
                        and fc.next_sibling.next_sibling == None):
                    sc = fc.next_sibling
                    # VP = VB* + PRT
                    if sc.tag == 'PRT':
                        self.__insert_after(WH, VP)
                        return SQ
                    # VP = VB* + PP
                    if sc.tag == 'PP':
                        ffc = sc.first_child
                        if ffc.tag == 'IN' and ffc.next_sibling == None:
                            self.__insert_after(WH, VP)
                            return SQ
                        if (ffc.tag == 'IN' and ffc.next_sibling != None
                                and ffc.next_sibling.next_sibling == None):
                            ssc = ffc.next_sibling
                            if ssc.tag in ('NP', 'ADJP'):
                                self.__insert_after(WH, fc)
                                return SQ
                    # VP = VB* + SBAR
                    if sc.tag == 'SBAR':
                        if fc.text in ('know', 'think'):
                            if WH.tag == 'WHADVP' and self.words[0] == 'how':
                                # WH = self.__prefix_by_to_WH(WH)
                                self.__insert_after(WH, VP)
                                return SQ
                            self.__insert_after(WH, VP)
                            return SQ
                        self.__insert_after(WH, fc)
                        return SQ
                    # VP = VB* + S
                    if sc.tag == 'S' and self.__tree_to_text(sc).startswith('(S(VP(TO to)(VP(VB'):
                        VB_S = sc.first_child.first_child.next_sibling.first_child
                        if VB_S.next_sibling == None:
                            self.__insert_after(WH, VP)
                            return SQ
                        if (VB_S.next_sibling.tag == 'SBAR' 
                                and VB_S.next_sibling.first_child.tag == 'WHADVP'):
                            self.__insert_after(WH, VB_S)
                            return SQ
                        self.__insert_after(WH, fc)
                        return SQ
                    # VP = VB* + ADVP
                    if sc.tag == 'ADVP':
                        if sc.first_child.text == 'atop':
                            self.__insert_after(WH, sc)
                        else:
                            self.__insert_after(WH, fc)
                        return SQ

                if WH.tag == 'WHADVP' and self.words[0] == 'how':
                    # WH = self.__prefix_by_to_WH(WH)
                    self.__insert_after(WH, VP)
                    return SQ
                self.__insert_after(WH, VP)
                return SQ

            # SQ = NP + VB* + NP
            if third.tag == 'NP':
                self.__insert_after(WH, third)
                return SQ
            # SQ = NP + VB* + S
            if third.tag == 'S' and self.__tree_to_text(third).startswith('(S(VP(TO to)(VP(VB'):
                VB_S = third.first_child.first_child.next_sibling.first_child
                if VB_S.next_sibling == None and WH.tag == 'WHNP':
                    self.__insert_after(WH, VB_S)
                    return SQ
                self.__insert_after(WH, second)
                return SQ
            # SQ = NP + VB* + SBAR
            if third.tag == 'SBAR' and third.first_child.tag == 'WHADVP':
                self.__insert_after(WH, second)
                return SQ
            # SQ = NP + VB* + PP
            if third.tag == 'PP':
                self.__insert_after(WH, third)
                return SQ
            # SQ = NP + VB* + ADJP
            if third.tag == 'ADJP':
                if WH.tag == 'WHADVP' and self.words[0] == 'how':
                    # WH = self.__prefix_by_to_WH(WH)
                    self.__insert_after(WH, third)
                    return SQ
                self.__insert_after(WH, third)
                return SQ

        raise ValueError('Unknown SQ structure!')
    
    def __adjust_SBARQ_question(self, WH, SQ):
        """Adjust word order of SBARQ question.

        Pipeline:
          1. __convert_WH_to_answer();
          2. __adjust_SQ_in_SBARQ();
          3. __insert_WH_into_SQ().
        """
        WH = self.__convert_WH_to_answer(WH)
        SQ = self.__adjust_SQ_in_SBARQ(SQ, WH)
        SQ = self.__insert_WH_into_SQ(WH, SQ)

        self.root.first_child.first_child = SQ
    
    def __adjust_S_question(self, S):
        prelast, last = S, S.first_child
        while last.next_sibling.tag != '.':
            prelast = last
            last = last.next_sibling
        words = ' '.join(last.gather_word())
        if 'who' in words:
            words = words.replace('who', 'the person')
        elif 'where' in words:
            words = words.replace('where', 'in the location')
        elif 'what' in words:
            words = words.replace('what', 'the')
        elif 'which' in words:
            words = words.replace('which', 'the')
        elif last.tag == 'VP':
            # (ROOT (S (NP (DT The) (NN kite)) (VP (VBZ is) (ADJP (JJ red) (CC or) (JJ blue))) (. ?)))
            if self.__check_VB(last.first_child):
                answer = self.__create_answer_node()
                self.__insert_after(answer, last.first_child)
                if not self.choices:
                    self.choices = self.__find_choices(answer.next_sibling)
                    if self.choices:
                        self.__delete_tree(answer, answer.next_sibling)
                    else:
                        raise ValueError()
                return
            else:
                raise ValueError('Unknown S structure! %s' % words)
        else:
            raise ValueError('Unknown S structure! %s' % words)
        self.__delete_tree(prelast, last)
        self.__insert_after(self.__create_answer_node(before_text=words), prelast)
        return
        
    def __prepare_SQ_answer(self, answer):
        answer = answer.lower()
        if self.choices:
            simplified = [list(filter(lambda w: w not in STOPWORDS, choice)) for choice in self.choices]
            simplified = list(map(lambda s: ' '.join(s), simplified))
            # assert answer in simplified, print(simplified, answer)
            if answer not in simplified:
                return False
            correct = simplified.index(answer)
            wrong = int(not correct)
            answer = ' '.join(self.choices[correct]) + ' rather than ' + ' '.join(self.choices[wrong])
        else:
            if answer not in ['yes', 'no']:
                return False
            # assert answer in ['yes', 'no']
            answer = "" if answer == 'yes' else 'not'
        return answer
    
    def __prepare_SBARQ_answer(self, answer):
        answer = answer.lower()
        if self.choices:
            simplified = [list(filter(lambda w: w not in STOPWORDS, choice)) for choice in self.choices]
            simplified = list(map(lambda s: ' '.join(s), simplified))
            # assert answer in simplified, print(simplified, answer)
            if answer not in simplified:
                return False
            correct = simplified.index(answer)
            wrong = int(not correct)
            answer = simplified[correct] + ' rather than ' + simplified[wrong]
        return answer
    
    def adjust_order(self):
        try:
            self.__replace_qmark_with_period()
            child = self.root.first_child
            assert child.next_sibling == None
            # Sometimes there are wrong parses
            if child.tag == "SINV":
                # (ROOT (SINV (WHPP (IN Behind) (WHNP (WHNP (WDT what) (NN kind)) (PP (IN of) (NP (NN furniture))))) (VBZ is) (NP (DT the) (NN window)) (. ?)))
                child.tag = "SBARQ"
                if child.first_child.next_sibling.tag != 'SQ':
                    assert child.first_child.tag in ('WHADJP', 'WHNP', 'WHADVP', 'WHPP')
                    SQ = self.Node(tag="SQ", text=None)
                    SQ.first_child = child.first_child.next_sibling
                    child.first_child.next_sibling = SQ
                    sqchild = SQ.first_child
                    while sqchild != None and sqchild.next_sibling.tag != '.':
                        sqchild = sqchild.next_sibling
                    assert sqchild != None
                    SQ.next_sibling = sqchild.next_sibling
                    sqchild.next_sibling = None
            if child.tag == "SQ" and \
                child.first_child.tag in ('WHADJP', 'WHNP', 'WHADVP', 'WHPP'):
                # (ROOT (SQ (WHPP (IN Behind) (WHNP (WDT what) (NN vehicle))) (SQ (VBP are) (NP (DT the) (NNS flowers))) (. ?)))
                child.tag = "SBARQ"
            if child.tag == "FRAG" and \
                child.first_child.tag in ('WHADJP', 'WHNP', 'WHADVP', 'WHPP'):
                # (ROOT (FRAG (WHPP (IN Of) (WHNP (WP what) (NN material))) (NP (DT the) (NN plate)) (. ?)))
                child.tag = "SBARQ"
            
            if child.tag == "SQ" and self.__check_VB(child.first_child):
                self.__adjust_SQ_question(child)
            elif child.tag == "SBARQ":
                prefirst = child
                first = child.first_child
                second = first.next_sibling
                if first.tag == 'SQ' and second == None:
                    self.__adjust_SQ_question(first)
                elif (first.tag in ('WHADJP', 'WHNP', 'WHADVP', 'WHPP')
                        and second.tag == 'SQ'):
                    # WHNP: what, which, who
                    # WHADJP: how
                    # WHPP: of what color
                    WH = self.__delete_tree(prefirst, first)
                    self.__adjust_SBARQ_question(WH, second)
                elif (first.tag in ('WHADJP', 'WHNP', 'WHADVP', 'WHPP')
                        and second.tag == 'S'):
                    second.tag = 'SQ'
                    WH = self.__delete_tree(prefirst, first)
                    self.__adjust_SBARQ_question(WH, second)
                elif (first.tag == 'ADVP' and \
                    second.tag in ('WHADJP', 'WHNP', 'WHADVP', 'WHPP') and \
                        second.next_sibling.tag == 'SQ'):
                    # (ROOT (SBARQ (ADVP (IN Inside)) (WHNP (WP what)) (SQ (VP (VBZ is) (NP (DT the) (NN pizza)))) (. ?))) 
                    ADVP = self.__delete_tree(prefirst, first)
                    WHNP = self.__delete_tree(prefirst, second)
                    WHPP = self.Node(tag='WHPP', text=None)
                    ADVP = self.__insert_as_first_child(ADVP, WHPP)
                    self.__insert_after(WHNP, ADVP)                    
                    self.__adjust_SBARQ_question(WHPP, prefirst.first_child)
                elif (first.tag in ('WHADJP', 'WHNP', 'WHADVP', 'WHPP')
                        and second.tag == 'VP'):
                        # (ROOT (SBARQ (WHNP (WHNP (WDT Which) (NN color)) (NP (NP (DT the) (NNS flowers)) (PP (IN in) (NP (NP (NN front)) (PP (IN of) (NP (DT the) (NN napkin))))))) (VP (VBP are)) (. ?)))
                    assert first.first_child.tag == "WHNP"
                    WH = self.__delete_tree(first, first.first_child)
                    WH = self.__convert_WH_to_answer(WH)
                    self.__insert_after(WH, second)
                elif (first.tag == "SBAR"
                        and second.tag == 'VP'):
                    # (ROOT (SBARQ (SBAR (WHNP (WP What)) (NP (NP (NP (DT the) (NNS items)) (PP (IN of) (NP (NN furniture)))) (SBAR (WHNP (WDT that)) (S (VP (VBP are) (ADJP (JJ empty))))))) (VP (VBP are) (VP (VBN called))) (. ?)))
                    assert first.first_child.tag in ('WHADJP', 'WHNP', 'WHADVP', 'WHPP')
                    WH = self.__delete_tree(first, first.first_child)
                    WH = self.__convert_WH_to_answer(WH)
                    self.__insert_after(WH, second)
                elif (first.tag in ('WHADJP', 'WHNP', 'WHADVP', 'WHPP')
                        and second.tag == 'NP'
                        and second.next_sibling.tag == 'VP'):
                    # (ROOT (SBARQ (WHNP (WP What)) (NP (NP (NP (DT the) (NN piece)) (PP (IN of) (NP (NN furniture)))) (SBAR (WHNP (WDT that)) (S (VP (VBZ is) (RB not) (ADJP (JJ comfortable)))))) (VP (VBZ is) (VP (VBN called))) (. ?)))
                    WH = self.__delete_tree(prefirst, first)
                    WH = self.__convert_WH_to_answer(WH)
                    self.__insert_after(WH, second.next_sibling)
                else:
                    raise ValueError('Unknown question structure!')
            elif child.tag == 'S':
                # (S (NP (DT The) (NN rug)) (VP (VBZ is) (PP (IN in) (NP (NP (NN front)) (PP (IN of) (NP (WP what)))))) (. ?))
                # (S (NP (DT The) (NN ladder)) (VP (VBZ is) (WHADVP (WRB where))) (. ?))
                # (ROOT(S(NP(PRP they))(VP(VBP are)(VP(VBG eating)(WHNP(WP what))))(. .)))
                self.__adjust_S_question(child)
            else:
                raise ValueError('Unknown question structure!')
            words = self.root.gather_word()
            words = filter(lambda w: w != '', words)
            statement = TreebankWordDetokenizer().detokenize(words)
            statement = statement[0].upper() + statement[1:]
            if ' - ' in statement:
                statement = statement.replace(' - ', '-')
            if 'called is the' in statement:
                statement = statement.replace('called is the', 'is called the')
            return statement
        except Exception as e:
            if DEBUG:
                print(self.question)
                print(self.root.tree_to_text())
                raise e
            else:
                return False
    
    @classmethod
    def prepare_answer(cls, answer, choices, sq=True):
        # answer: str
        # choices: list of str
        answer = answer.lower()
        answer = list(filter(lambda w: w not in STOPWORDS, answer.split()))
        answer = ' '.join(answer)
        if choices:
            choices = [choice.lower() for choice in choices]
            simplified = [list(filter(lambda w: w not in STOPWORDS, choice.split())) for choice in choices]
            simplified = list(map(lambda s: ' '.join(s), simplified))
            # assert answer in simplified, print(simplified, answer)
            if answer not in simplified:
                # What kind of furniture is modern, the cabinets of the bathroom or the shelves?
                for j, s in enumerate(simplified):
                    w = s.split()
                    if answer not in w:
                        return False
                    else:
                        correct = j
                        break
            else:
                correct = simplified.index(answer)
            wrong = int(not correct)
            if answer in ['yes', 'no']:
                answer = '' if answer == 'yes' else 'not'
            else:
                if sq:
                    answer = choices[correct] + ' rather than ' + choices[wrong]
                else:
                    answer = simplified[correct] + ' rather than ' + simplified[wrong]
        return answer
        
    def replace_answer(self, answer, statement=None, prepare=True):
        """
        answer: prepared or unprepared answer
        statement: (optional) statement with **blank** in it
        prepare: if true, prepare the answer

        return: statement with answer in it or false if error
        """
        try:
            if prepare:
                child = self.root.first_child
                if child.tag == "SQ":
                    answer = self.__prepare_SQ_answer(answer)
                elif child.tag in ["SBARQ", "S"]:
                    answer = self.__prepare_SBARQ_answer(answer)                
                else:
                    raise ValueError('Unknown question structure!')
            if statement is None:
                words = self.root.gather_word()
                words = filter(lambda w: w != '', words)
                statement = TreebankWordDetokenizer().detokenize(words)
                statement = statement[0].upper() + statement[1:]
                if ' - ' in statement:
                    statement = statement.replace(' - ', '-')
                if 'called is the' in statement:
                    statement = statement.replace('called is the', 'is called the')
            statement = statement.replace("**blank**", answer)
            return statement
        except Exception as e:
            if DEBUG:
                print(self.question)
                print(self.root.tree_to_text())
                raise e
            else:
                return False
    

def check_answer_valid(answer, choices):
    # index if the answer is valid, false otherwise
    answer = answer.lower()
    answer = list(filter(lambda w: w not in STOPWORDS, answer.split()))
    answer = ' '.join(answer)
    choices = [choice.lower() for choice in choices]
    simplified = [list(filter(lambda w: w not in STOPWORDS, choice.split())) for choice in choices]
    simplified = list(map(lambda s: ' '.join(s), simplified))
    if answer not in simplified:
        # What kind of furniture is modern, the cabinets of the bathroom or the shelves?
        for j, s in enumerate(simplified):
            w = s.split()
            if answer not in w:
                return False
            else:
                return j
    else:
        return simplified.index(answer)

if __name__ == "__main__":
    while True:
        print("Enter question: ")
        sentence = input()
        tree = POSTree(sentence.strip())
        print(tree.adjust_order())
        print(tree.choices)
        # print("Enter answer: ")
        # answer = input()
        # print(tree.replace_answer(answer.strip()))