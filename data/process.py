import json
import random
import pdb
import re
import os
import sys
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import itertools
import copy

all_speaker2convs = defaultdict(dict)
unseen_user_pairs = defaultdict(list)
unseen_conv_pairs = defaultdict(list)
unseen_user_hard_pairs = defaultdict(list)
unseen_conv_hard_pairs = defaultdict(list)
unseen_user_harder_pairs = defaultdict(list)
unseen_conv_harder_pairs = defaultdict(list)
seen_conv_pairs = defaultdict(list)
seen_conv_hard_pairs = defaultdict(list)
seen_conv_harder_pairs = defaultdict(list)
train_pairs = defaultdict(list)
dev_pairs = defaultdict(list)
all_unseen_conv_speaker2convs = defaultdict(dict)
all_unseen_users = defaultdict(list)
all_seen_users = defaultdict(list)


# We ignore the roles who is not a typical roles in dataset, such as 'Man On TV'.
ignore_roles = [
    '#ALL#', 'Woman', 'Guy', 'Man', 'Waiter', 'The Director', 'Director', 'Both', 'Nurse', 'Receptionist', 'The Guys', 'Girl', 'The Interviewer', 'The Girls', 'Waitress', 'Customer', 'Doctor', 'Announcer', 'Guys', 'The Casting Director', "Bob (Chandler's coworker)", 'Paul the Wine Guy', 'Kid', 'Interviewer', 'Big Bully', 'Little Bully', 'The Salesman', 
    "Joey's Doctor", 'The Teacher', 'The Woman', 'Fireman No. 3', 'Intercom', 'Fake Monica', 'Host', 'Student', 'Julio (poet)', 'Hold Voice', 'Stage Manager', 'Voice', 'The Waiter', 'The Man', 'C.H.E.E.S.E.', 'The Dry Cleaner', 'The Museum Official', 'The Cooking Teacher', 'Fireman No. 1', 'Fireman No. 2', 'Steve (drug addict)', 'Woman No. 1', 'Woman No. 2', 
    "Director's Assistant", 'Friend', "Joey's Co-Star", 'Eric (photographer)' 'Friend No. 1', 'Friend No. 2', "Joey's Sisters", "Richard's Date", 'Singer', 'Lecturer', 'Tv Announcer', 'The Conductor', 'Tour Guide', 'Ticket Counter Attendant', 'The Singing Man', 'The Stripper', 'Passenger', 'Gate Agent', 'Stage Director', 'Cop', 'Answering Machine', 'The Croupier', 
    'The Flight Attendant', "Joey's Hand Twin", 'The Lurker', 'The Security Guard', 'The Potential Roommate', 'Tall Guy', "Carl (Joey's lookalike)", 'Professor Feesen', 'Male Jeweler', 'The Cute Guy', "Joey's Date", 'Supervisor', 'Policeman', 'The Assistant Director', 'The Rabbi', 'Photographer', 'Bandleader', 'Front Desk Clerk', 'Hooker', "Mrs. Verhoeven's Daughter", 
    'Man On Tv', 'Woman On Tv', 'Katie (saleswoman)', 'Others', 'Intern', 'Message', 'Professor Spafford', 'Precious', 'Air Stewardess', 'Passenger #2', 'Alice Knight', 'Charity Guy', 'Delivery Guy', 'Eric (photographer)', 'Friend No. 1', 'Gang', 'Girl 1', 'Grandma Tribbiani', 'Grandmother', 'Hombre Man', 'Hypnosis Tape', 'Jade', 'Kids', 'Machine', 'Pizza Guy', 'Quartet',
    'Realtor', 'Security Guard', 'Teacher', 'The Doctor', 'Ticket Agent', 'Strangers 1', 'Strangers 2', 'classmates', 'stranger', 'wizard', 'Man', 'Students', 'Other', 'Voice', 'Man', 'Girl', 'Missy', 'Woman', 'Nurse', 'Assistant', 'All', 'Waiter'
]
main_roles = ['Chandler Bing', 'Rachel Green', 'Joey Tribbiani', 'Harry Potter', 'Hermione Granger', 'Ron Weasley', 'Voldemort' 'Sheldon', 'Penny', 'Leonard']

def utterance_process(utterances, conversation):
    utters = []
    for u in utterances.split('\n'):
        if len(u) > 0:
            utters.append(u)
    if len(' '.join(utters).split()) < 50 or len(utters) < 5:
        return None, None
    conv = []
    for u in conversation.split('\n'):
        if len(u) > 0:
            us = u.split(': ')
            conv.append({'speaker': us[0], 'utterance': ': '.join(us[1:])})
    return utters, conv

def get_negative(sample):
    negative_samples = []
    anchor_speaker = sample['target_speaker']
    speaker2utters = defaultdict(list)
    for utter in sample['conversation']:
        if utter['speaker'] != anchor_speaker and utter['speaker'] not in ignore_roles:
            speaker2utters[utter['speaker']].append(utter['utterance'])
    for speaker in list(speaker2utters.keys()):
        if len(' '.join(speaker2utters[speaker]).split()) < 50 or len(speaker2utters[speaker]) < 5:
            continue
        negative_samples.append(
            {'dataset': sample['dataset'], 'target_speaker': speaker, 'utterances': speaker2utters[speaker], 'conversation': sample['conversation']}
        )
    return negative_samples

def unseen_user(speakers_dict, speakers_list):
    pairs = []
    for speaker in tqdm(speakers_list, 'unseen_user_hard'):
        utterances = speakers_dict[speaker]
        for anchor in utterances:
            positive = random.choice([utter for utter in utterances if utter != anchor])
            negative_speaker = random.choice([spk for spk in speakers_list if spk != speaker])
            negative = random.choice(speakers_dict[negative_speaker])
            pairs.append((anchor, positive, 1))
            pairs.append((anchor, negative, 0))
    return pairs

def unseen_user_hard(speakers_dict, speakers_list):
    pairs = []
    for speaker in tqdm(speakers_list, 'unseen_user_hard'):
        utterances = speakers_dict[speaker]
        for anchor in utterances:
            positive = random.choice([utter for utter in utterances if utter != anchor])
            negative_speaker = random.choice([spk for spk in speakers_list if spk != speaker])
            negative = random.choice(speakers_dict[negative_speaker])
            pairs.append((anchor, positive, 1))
            pairs.append((anchor, negative, 0))
    return pairs

def unseen_user_harder(speakers_dict, speakers_list):
    pairs = []
    for speaker in tqdm(speakers_list, 'unseen_user_harder'):
        samples = speakers_dict[speaker]
        for anchor in samples:
            negatives = get_negative(anchor)
            if len(negatives) < 1:
                continue
            negative = random.choice(negatives)
            positive = random.choice([sample for sample in samples if sample!=anchor])
            pairs.append((anchor, positive, 1))
            pairs.append((anchor, negative, 0))
    return pairs

def unseen_conv(speakers_dict, seen_speaker_dict):
    pairs = []
    speakers_list = list(speakers_dict.keys())
    for speaker in tqdm(speakers_list, 'unseen_conv'):
        utterances = speakers_dict[speaker]
        for anchor in utterances:
            positive = random.choice(seen_speaker_dict[speaker])
            negative_speaker = random.choice([spk for spk in speakers_list if spk != speaker])
            negative = random.choice(seen_speaker_dict[negative_speaker])
            pairs.append((anchor, positive, 1))
            pairs.append((anchor, negative, 0))
    return pairs

def unseen_conv_hard(speakers_dict, seen_speaker_dict):
    pairs = []
    speakers_list = list(speakers_dict.keys())
    for speaker in tqdm(speakers_list, 'unseen_conv_hard'):
        utterances = speakers_dict[speaker]
        for anchor in utterances:
            positive = random.choice(seen_speaker_dict[speaker])
            negative_speaker = random.choice([spk for spk in speakers_list if spk != speaker])
            negative = random.choice(seen_speaker_dict[negative_speaker])
            pairs.append((anchor, positive, 1))
            pairs.append((anchor, negative, 0))
    return pairs

def unseen_conv_harder(speakers_dict, seen_speaker_dict):
    pairs = []
    speakers_list = list(speakers_dict.keys())
    for speaker in tqdm(speakers_list, 'unseen_user_harder'):
        samples = speakers_dict[speaker]
        for anchor in samples:
            negatives = get_negative(anchor)
            if len(negatives) < 1:
                continue
            negative = random.choice(negatives)
            positive = random.choice(seen_speaker_dict[speaker])
            pairs.append((anchor, positive, 1))
            pairs.append((anchor, negative, 0))
    return pairs

def train_data_harder(speakers_dict, speakers_list):
    pairs = []
    for speaker in tqdm(speakers_list, 'train'):
        samples = speakers_dict[speaker]
        for anchor in samples:
            negatives = get_negative(anchor)
            for negative in negatives:
                positive = random.choice([sample for sample in samples if sample!=anchor])
                pairs.append((anchor, positive, 1))
                pairs.append((anchor, negative, 0))
    return pairs

def train_data_hard(speakers_dict, speakers_list):
    pairs = []
    for speaker in tqdm(speakers_list, 'train_hard'):
        utterances = speakers_dict[speaker]
        for anchor in utterances:
            positive = random.choice([utter for utter in utterances if utter != anchor])
            pairs.append((anchor, positive, 1))
            for _ in range(3):
                negative_speaker = random.choice([spk for spk in speakers_list if spk != speaker])
                negative = random.choice(speakers_dict[negative_speaker])
                pairs.append((anchor, negative, 0))
    return pairs

def train_data(speakers_dict, speakers_list):
    pairs = []
    for speaker in tqdm(speakers_list, 'train'):
        utterances = speakers_dict[speaker]
        for anchor in utterances:
            positive = random.choice([utter for utter in utterances if utter != anchor])
            pairs.append((anchor, positive, 1))
            for _ in range(3):
                negative_speaker = random.choice([spk for spk in speakers_list if spk != speaker])
                negative = random.choice(speakers_dict[negative_speaker])
                pairs.append((anchor, negative, 0))
    return pairs

def process_n_fold_per_dataset(speaker2convs, folds):
    speakers = list(speaker2convs.keys())
    random.shuffle(speakers)
    for fold_idx in range(folds):
        fold_speaker2convs = copy.deepcopy(speaker2convs)
        candidates = [s for s in speakers if len(speaker2convs[s]) < 50 and s not in main_roles]
        unseen_speakers = random.sample(candidates, min(5, len(candidates)//3))
        all_unseen_users[fold_idx].extend(unseen_speakers)
        seen_speakers = [s for s in speakers if s not in unseen_speakers]
        all_seen_users[fold_idx].extend(seen_speakers)
        unseen_user_hard_set = unseen_user_hard(fold_speaker2convs, unseen_speakers)
        unseen_user_harder_set = unseen_user_harder(fold_speaker2convs, unseen_speakers)
        unseen_conv_speaker2convs = {}
        for speaker in seen_speakers:
            if len(fold_speaker2convs[speaker]) < 20:
                sample_num = 1
            else:
                sample_num = 2
            unseen_conv_speaker2convs[speaker] = []
            for _ in range(sample_num):
                idx = random.randint(0, len(fold_speaker2convs[speaker])-1)
                unseen_conv_speaker2convs[speaker].append(fold_speaker2convs[speaker][idx])
                fold_speaker2convs[speaker].pop(idx)
        all_speaker2convs[fold_idx].update(fold_speaker2convs)
        unseen_conv_hard_set = unseen_conv_hard(unseen_conv_speaker2convs, fold_speaker2convs)
        unseen_conv_harder_set = unseen_conv_harder(unseen_conv_speaker2convs, fold_speaker2convs)
        all_unseen_conv_speaker2convs[fold_idx].update(unseen_conv_speaker2convs)
        train_set_hard = train_data_hard(fold_speaker2convs, seen_speakers)
        train_set_harder = train_data_harder(fold_speaker2convs, seen_speakers)
        seen_harder_set = random.sample(train_set_harder, len(train_set_harder)//20)
        seen_hard_set = random.sample(train_set_hard, len(train_set_hard)//20)
        unseen_user_hard_pairs[fold_idx].extend(unseen_user_hard_set)
        unseen_user_harder_pairs[fold_idx].extend(unseen_user_harder_set)
        unseen_conv_hard_pairs[fold_idx].extend(unseen_conv_hard_set)
        unseen_conv_harder_pairs[fold_idx].extend(unseen_conv_harder_set)
        seen_conv_hard_pairs[fold_idx].extend(seen_hard_set)
        seen_conv_harder_pairs[fold_idx].extend(seen_harder_set)
        train_pairs[fold_idx].extend(train_set_hard)
        train_pairs[fold_idx].extend(train_set_harder)
        dev_pairs[fold_idx].extend(random.sample(unseen_user_hard_set, min(50, len(unseen_user_hard_set))))
        dev_pairs[fold_idx].extend(random.sample(unseen_user_harder_set, min(50, len(unseen_user_harder_set))))
        dev_pairs[fold_idx].extend(random.sample(unseen_conv_hard_set, min(50, len(unseen_conv_hard_set))))
        dev_pairs[fold_idx].extend(random.sample(unseen_conv_harder_set, min(50, len(unseen_conv_harder_set))))

# CMD
characters = {}
with open('./CMD/movie_characters_metadata.txt', encoding='latin-1') as f:
    lines = f.readlines()
    for l in lines:
        metas = l.strip().split(' +++$+++ ')
        characters[metas[0]] = metas[1]

utterances = {}
with open('./CMD/movie_lines.txt', encoding='latin-1') as f:
    lines = f.readlines()
    for l in lines:
        metas = l.strip().split(' +++$+++ ')
        utterances[metas[0]] = {'speaker': metas[1], 'utterance': metas[-1]}

movie2title = {}
with open('./CMD/movie_titles_metadata.txt', encoding='latin-1') as f:
    lines = f.readlines()
    for l in lines:
        metas = l.strip().split(' +++$+++ ')
        movie2title[metas[0]] = metas[1]

movie2convs = {}
with open('./CMD/movie_conversations.txt', encoding='latin-1') as f:
    lines = f.readlines()
    last_uid = 0
    utters = []
    speakers = []
    for l in lines:
        metas = l.strip().split(' +++$+++ ')
        uids = eval(metas[-1])
        if int(uids[0][1:]) - 1 != last_uid:
            conversation = {'speakers': speakers, 'utterances': utters}
            title = movie2title[metas[2]]
            if title not in movie2convs:
                movie2convs[title] = []
            if len(conversation['speakers']) > 10:
                movie2convs[title].append(conversation)
            utters = []
            speakers = []
        for uid in uids:
            utters.append(utterances[uid]['utterance'])
            speakers.append(characters[utterances[uid]['speaker']])
        last_uid = int(uids[-1][1:])

speaker2convs = {}
for movie, convs in movie2convs.items():
    speaker2convs_movie = defaultdict(list)
    for conv in convs:
        speakers = []
        utterances = []
        conversation = ''
        for speaker, utter in zip(conv['speakers'], conv['utterances']):
            if '</' in utter:
                continue
            conversation += f'{speaker}: {utter}\n'
            if speaker not in speakers:
                speakers.append(speaker)
                utterances.append(utter + '\n')
            else:
                index = speakers.index(speaker)
                utterances[index] += utter + '\n'
        for speaker, utters in zip(speakers, utterances):
            utters, conv = utterance_process(utters, conversation)
            if utters:
                speaker2convs_movie[speaker].append({'dataset': 'CMD', 'target_speaker': speaker, 'utterances': utters, 'conversation': conv})
    for speaker in list(speaker2convs_movie.keys()):
        speaker2convs[speaker] = speaker2convs_movie[speaker]


# Friends
movie2convs = {}
for season in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']:
    movie2convs[season] = []
    with open(f'./Friends/friends_season_{season}.json') as f:
        d = json.load(f)
        episodes = d['episodes']
        for episode in episodes:
            scenes = episode['scenes']
            for scene in scenes:
                conversation = {'speakers': [], 'utterances': []}
                utterances = scene['utterances']
                if len(utterances) < 5:
                    continue
                for utter in utterances:
                    if len(utter['speakers']) < 1:
                        continue
                    for speaker in utter['speakers']:
                        conversation['speakers'].append(speaker)
                        conversation['utterances'].append(utter['transcript'])
                movie2convs[season].append(conversation)

speaker2convs = {}
for movie_id, convs in movie2convs.items():
    for conv in convs:
        speakers = []
        utters = []
        original_conversation = ''
        for speaker, utterance in zip(conv['speakers'], conv['utterances']):
            if speaker == 'Cailin':
                speaker = 'Caitlin'
            if speaker == 'Celia':
                speaker = 'Cecilia'
            original_conversation += f'{speaker}: {utterance}\n'
            if speaker not in speakers:
                speakers.append(speaker)
                utters.append('')
            utters[speakers.index(speaker)] += utterance + '\n'
        for speaker, utter in zip(speakers, utters):
            if speaker not in speaker2convs:
                speaker2convs[speaker] = []
            utters, conv = utterance_process(utter, original_conversation)
            if utters:
                speaker2convs[speaker].append(
                    {
                        'dataset': 'Friends',
                        'target_speaker': speaker,
                        'utterances': utters,
                        'conversation': conv,
                    }
                )
for speaker in list(speaker2convs.keys()):
    if speaker in ['#ALL#', 'Woman', 'Guy', 'Man', 'Waiter', 'The Director', 'Director', 'Both', 'Nurse', 'Receptionist', 'The Guys', 'Girl', 'The Interviewer', 'The Girls', 'Waitress', 'Customer', 'Doctor', 'Announcer', 'Guys', 'The Casting Director', "Bob (Chandler's coworker)", 'Paul the Wine Guy', 'Kid', 'Interviewer', 'Big Bully', 'Little Bully', 'The Salesman', 
                   "Joey's Doctor", 'The Teacher', 'The Woman', 'Fireman No. 3', 'Intercom', 'Fake Monica', 'Host', 'Student', 'Julio (poet)', 'Hold Voice', 'Stage Manager', 'Voice', 'The Waiter', 'The Man', 'C.H.E.E.S.E.', 'The Dry Cleaner', 'The Museum Official', 'The Cooking Teacher', 'Fireman No. 1', 'Fireman No. 2', 'Steve (drug addict)', 'Woman No. 1', 'Woman No. 2', 
                   "Director's Assistant", 'Friend', "Joey's Co-Star", 'Eric (photographer)' 'Friend No. 1', 'Friend No. 2', "Joey's Sisters", "Richard's Date", 'Singer', 'Lecturer', 'Tv Announcer', 'The Conductor', 'Tour Guide', 'Ticket Counter Attendant', 'The Singing Man', 'The Stripper', 'Passenger', 'Gate Agent', 'Stage Director', 'Cop', 'Answering Machine', 'The Croupier', 
                   'The Flight Attendant', "Joey's Hand Twin", 'The Lurker', 'The Security Guard', 'The Potential Roommate', 'Tall Guy', "Carl (Joey's lookalike)", 'Professor Feesen', 'Male Jeweler', 'The Cute Guy', "Joey's Date", 'Supervisor', 'Policeman', 'The Assistant Director', 'The Rabbi', 'Photographer', 'Bandleader', 'Front Desk Clerk', 'Hooker', "Mrs. Verhoeven's Daughter", 
                   'Man On Tv', 'Woman On Tv', 'Katie (saleswoman)', 'Others', 'Intern', 'Message', 'Professor Spafford', 'Precious', 'Air Stewardess', 'Passenger #2', 'Alice Knight', 'Charity Guy', 'Delivery Guy', 'Eric (photographer)', 'Friend No. 1', 'Gang', 'Girl 1', 'Grandma Tribbiani', 'Grandmother', 'Hombre Man', 'Hypnosis Tape', 'Jade', 'Kids', 'Machine', 'Pizza Guy', 'Quartet',
                   'Realtor', 'Security Guard', 'Teacher', 'The Doctor', 'Ticket Agent']:
        speaker2convs.pop(speaker)

for speaker in list(speaker2convs.keys()):
    if len(speaker2convs[speaker]) < 5:
        speaker2convs.pop(speaker)

process_n_fold_per_dataset(speaker2convs, 3)

# HPD
movie2convs = {}
with open('./HPD/en_test_set.json') as f:
    test = json.load(f)
    for session in list(test.values()):
        conversation = {'speakers': [], 'utterances': []}
        season = int(session['position'][4:5])
        if season not in movie2convs:
            movie2convs[season] = []
        if len(session['dialogue']) < 5:
            continue
        for utter in session['dialogue']:
            utters = utter.split(': ')
            speaker = utters[0].lstrip().strip()
            utterance = ' '.join(utters[1:])
            conversation['speakers'].append(speaker)
            conversation['utterances'].append(utterance)
        if 'positive_response' in session:
            conversation['speakers'].append('Harry')
            conversation['utterances'].append(session['positive_response'])
        movie2convs[season].append(conversation)
with open('./HPD/en_train_set.json') as f:
    train = json.load(f)
    for session in list(train.values()):
        conversation = {'speakers': [], 'utterances': []}
        season = int(session['position'][4:5])
        if season not in movie2convs:
            movie2convs[season] = []
        if len(session['dialogue']) < 5:
            continue
        for utter in session['dialogue']:
            utters = utter.split(': ')
            speaker = utters[0].lstrip().strip()
            utterance = ' '.join(utters[1:])
            conversation['speakers'].append(speaker)
            conversation['utterances'].append(utterance)
        if 'positive_response' in session:
            conversation['speakers'].append('Harry')
            conversation['utterances'].append(session['positive_response'])
        movie2convs[season].append(conversation)
speaker2name = {
    'Vernon': 'Vernon Dursley',
    'Harry': 'Harry Potter',
    'Hagrid': 'Rubeus Hagrid',
    'Snape': 'Severus Snape',
    'Hermione': 'Hermione Granger',
    'Ron': 'Ron Weasley',
    'McGonagall': 'Minerva McGonagall',
    'Malfoy': 'Draco Malfoy',
    'Dumbledore': 'Albus Dumbledore',
    'Weasley': 'Arthur Weasley',
    'Mrs. Weasley': 'Molly Weasley',
    'Fudge': 'Cornelius Fudge',
    'Lupin': 'Remus Lupin',
    'Sirius': 'Sirius Black',
    'Moody': 'Alastor Moody',
    'Umbridge': 'Dolores Umbridge',
    'Bellatrix': 'Bellatrix Lestrange',
    'Slughorn': 'Horace Slughorn'
}
speaker2convs = {}
for movie_id, convs in movie2convs.items():
    for conv in convs:
        speakers = []
        utters = []
        original_conversation = ''
        for speaker, utterance in zip(conv['speakers'], conv['utterances']):
            original_conversation += f'{speaker}: {utterance}\n'
            if speaker not in speakers:
                if speaker in speaker2name:
                    name = speaker2name[speaker]
                else:
                    name = speaker
                speakers.append(name)
                utters.append('')
            utters[speakers.index(name)] += utterance + '\n'
        for speaker, utter in zip(speakers, utters):
            if speaker not in speaker2convs:
                speaker2convs[speaker] = []
            utters, conv = utterance_process(utter, original_conversation)
            if utters:
                speaker2convs[speaker].append(
                    {
                        'dataset': 'HarryPotter',
                        'target_speaker': speaker,
                        'utterances': utters,
                        'conversation': conv,
                    }
                )

for speaker in list(speaker2convs.keys()):
    if speaker in ['Strangers 1', 'Strangers 2', 'classmates', 'stranger', 'wizard']:
        speaker2convs.pop(speaker)

# HarryPotterMovies
characters = pd.read_csv('./Harry_Potter_Movies/Characters.csv', encoding='ISO-8859-1')
json_data = json.loads(characters.to_json(orient='records'))
id2character = {}
for character in json_data:
    id2character[character['Character ID']] = character
utters = pd.read_csv('./Harry_Potter_Movies/Dialogue.csv', encoding='ISO-8859-1')
utters = json.loads(utters.to_json(orient='records'))
all_convs = []
conv = {'speakers': [], 'utterances': []}
chapter = 0
place = 0
for utter in utters:
    if utter['Chapter ID'] != chapter or utter['Place ID'] != place:
        if len(conv['speakers']) > 5:
            all_convs.append(conv)
        conv = {'speakers': [], 'utterances': []}
        chapter = utter['Chapter ID']
        place = utter['Place ID']
    speaker = id2character[utter['Character ID']]['Character Name']
    utterance = utter['Dialogue']
    conv['speakers'].append(speaker)
    conv['utterances'].append(utterance)
for conv in all_convs:
    speakers = []
    utters = []
    original_conversation = ''
    for speaker, utterance in zip(conv['speakers'], conv['utterances']):
        original_conversation += f'{speaker}: {utterance}\n'
        if speaker == 'Tom Riddle':
            speaker = 'Voldemort'
        if speaker not in speakers:
            speakers.append(speaker)
            utters.append('')
        utters[speakers.index(speaker)] += utterance + '\n'
    for speaker, utter in zip(speakers, utters):
        if speaker not in speaker2convs:
            speaker2convs[speaker] = []
        utters, conv = utterance_process(utter, original_conversation)
        if utters:
            speaker2convs[speaker].append(
                {
                    'dataset': 'HarryPotter',
                    'target_speaker': speaker,
                    'utterances': utters,
                    'conversation': conv,
                }
            )

speakers = []
for speaker in list(speaker2convs.keys()):
    if speaker in ['Man', 'Students', 'Other']:
        speaker2convs.pop(speaker)

for speaker in list(speaker2convs.keys()):
    if len(speaker2convs[speaker]) < 5:
        speaker2convs.pop(speaker)

process_n_fold_per_dataset(speaker2convs, 3)

# TheBigBang
with open('./TheBigBang/all.txt') as f:
    lines = f.readlines()
all_convs = []
conv = {'speakers': [], 'utterances': []}
for l in lines:
    l = l.strip()
    if len(l) == 0:
        if len(conv['speakers']) > 5:
            all_convs.append(conv)
        conv = {'speakers': [], 'utterances': []}
    else:
        metas = l.split(':')
        speaker = metas[0]
        text = ':'.join(metas[1:]).strip().lstrip()
        conv['speakers'].append(speaker)
        conv['utterances'].append(text)

speaker2convs = {}
for conv in all_convs:
    speakers = []
    utters = []
    original_conversation = ''
    for speaker, utterance in zip(conv['speakers'], conv['utterances']):
        original_conversation += f'{speaker}: {utterance}\n'
        if speaker == 'Rajj':
            speaker = 'Raj'
        if speaker not in speakers:
            speakers.append(speaker)
            utters.append('')
        utters[speakers.index(speaker)] += utterance + '\n'
    for speaker, utter in zip(speakers, utters):
        if speaker not in speaker2convs:
            speaker2convs[speaker] = []
        utters, conv = utterance_process(utter, original_conversation)
        if utters:
            speaker2convs[speaker].append(
                {
                    'dataset': 'TheBigBangTheory',
                    'target_speaker': speaker,
                    'utterances': utters,
                    'conversation': conv,
                }
            )

speakers = []
for speaker in list(speaker2convs.keys()):
    if speaker in ['Voice', 'Man', 'Girl', 'Missy', 'Woman', 'Nurse', 'Assistant', 'All', 'Waiter']:
        speaker2convs.pop(speaker)

for speaker in list(speaker2convs.keys()):
    if len(speaker2convs[speaker]) < 5:
        speaker2convs.pop(speaker)

process_n_fold_per_dataset(speaker2convs, 3)


# MSC
data_list = []
with open('./MSC/test.jsonl') as f:
    lines = f.readlines()
    for l in lines:
        data_list.append(json.loads(l))
with open('./MSC/train.jsonl') as f:
    lines = f.readlines()
    for l in lines:
        data_list.append(json.loads(l))
with open('./MSC/valid.jsonl') as f:
    lines = f.readlines()
    for l in lines:
        data_list.append(json.loads(l))
speaker2convs = {}
for idx in range(len(data_list)):
    dialog = data_list[idx]
    speaker2convs[f'Dialog {idx}-Speaker 1'] = []
    speaker2convs[f'Dialog {idx}-Speaker 2'] = []
    for session in dialog['sessions']:
        session1, session2 = "", ""
        total_session = ""
        for utter in session['dialogue']:
            if utter['speaker'] == 'Speaker 1':
                session1 += utter['text'] + '\n'
                total_session += f'Dialog {idx}-Speaker 1' + ': ' + utter['text'] + '\n'
            elif utter['speaker'] == 'Speaker 2':
                session2 += utter['text'] + '\n'
                total_session += f'Dialog {idx}-Speaker 2' + ': ' + utter['text'] + '\n'
        utters, conv = utterance_process(session1, total_session)
        if utters:
            speaker2convs[f'Dialog {idx}-Speaker 1'].append({'dataset': 'MSC','target_speaker': f'Dialog {idx}-Speaker 1', 'utterances': utters, 'conversation': conv})
        utters, conv = utterance_process(session2, total_session)
        if utters:
            speaker2convs[f'Dialog {idx}-Speaker 2'].append({'dataset': 'MSC', 'target_speaker': f'Dialog {idx}-Speaker 2', 'utterances': utters, 'conversation': conv})

for speaker in list(speaker2convs.keys()):
    if len(speaker2convs[speaker]) < 3:
        speaker2convs.pop(speaker)

process_n_fold_per_dataset(speaker2convs, 3)

# AnnoMI
data_list = []
with open('./AnnoMI/high.jsonl') as f:
    lines = f.readlines()
    for l in lines:
        sample = json.loads(l)
        data_list.append(sample)

with open('./AnnoMI/low.jsonl') as f:
    lines = f.readlines()
    for l in lines:
        sample = json.loads(l)
        data_list.append(sample)


def split_list_into_sublists(utterances, users):
    sublists_utter = []
    sublists_user = []
    while len(utterances) > 0:
        length = random.randint(10, 20)
        sublist_utter = utterances[:length]
        sublist_user = users[:length]
        if sublist_user.count('client') < 3:
            sublists_utter[-1] += sublist_utter
            sublists_user[-1] += sublist_user
        sublists_utter.append(sublist_utter)
        sublists_user.append(sublist_user)
        utterances = utterances[length:]
        users = users[length:]
    return sublists_utter, sublists_user

speaker2convs = {}
for idx in range(len(data_list)):
    d = data_list[idx]
    topic = d['topic']
    speaker2convs[f'AnnoMI-{idx}'] = []
    utterances_list, speakers_list = split_list_into_sublists(d['utterances'], d['interlocutors'])
    for speakers, utters in zip(speakers_list, utterances_list):
        utterances = ""
        conversation = ""
        for speaker, utter in zip(speakers, utters):
            if speaker == 'client':
                utterances += utter + '\n'
            conversation += f"{speaker}-{idx}: {utter}\n"
        utters, conv = utterance_process(utterances, conversation)
        if utters:
            speaker2convs[f'AnnoMI-{idx}'].append({'dataset': 'AnnoMI', 'topic': topic, 'target_speaker': f'client-{idx}', 'utterances': utters, 'conversation': conv})

for speaker in list(speaker2convs.keys()):
    if len(speaker2convs[speaker]) < 5:
        speaker2convs.pop(speaker)

process_n_fold_per_dataset(speaker2convs, 3)

for fold in range(3):
    with open(f'../output/train_{fold}.jsonl', 'w') as f:
        for triple in train_pairs[fold]:
            anchor, sample, label = triple
            f.write(
                json.dumps(
                    {
                        'anchor_dataset': anchor['dataset'],
                        'anchor_speaker': anchor['target_speaker'],
                        'anchor_utterances': anchor['utterances'],
                        'anchor_conversation': anchor['conversation'],
                        'paired_dataset': sample['dataset'],
                        'paired_speaker': sample['target_speaker'],
                        'paired_utterances': sample['utterances'],
                        'paired_conversation': sample['conversation'],
                        'label': label,
                    }, ensure_ascii=False
                ) + '\n'
            )

    with open(f'../output/dev_{fold}.jsonl', 'w') as f:
        for triple in dev_pairs[fold]:
            anchor, sample, label = triple
            f.write(
                json.dumps(
                    {
                        'anchor_dataset': anchor['dataset'],
                        'anchor_speaker': anchor['target_speaker'],
                        'anchor_utterances': anchor['utterances'],
                        'anchor_conversation': anchor['conversation'],
                        'paired_dataset': sample['dataset'],
                        'paired_speaker': sample['target_speaker'],
                        'paired_utterances': sample['utterances'],
                        'paired_conversation': sample['conversation'],
                        'label': label,
                    }, ensure_ascii=False
                ) + '\n'
            )

    with open(f'../output/seen_test_base_{fold}.jsonl', 'w') as f:
        for triple in seen_conv_pairs[fold]:
            anchor, sample, label = triple
            f.write(
                json.dumps(
                    {
                        'anchor_dataset': anchor['dataset'],
                        'anchor_speaker': anchor['target_speaker'],
                        'anchor_utterances': anchor['utterances'],
                        'anchor_conversation': anchor['conversation'],
                        'paired_dataset': sample['dataset'],
                        'paired_speaker': sample['target_speaker'],
                        'paired_utterances': sample['utterances'],
                        'paired_conversation': sample['conversation'],
                        'label': label,
                    }, ensure_ascii=False
                ) + '\n'
            )

    with open(f'../output/unseen_conv_base_{fold}.jsonl', 'w') as f:
        for triple in unseen_conv_pairs[fold]:
            anchor, sample, label = triple
            f.write(
                json.dumps(
                    {
                        'anchor_dataset': anchor['dataset'],
                        'anchor_speaker': anchor['target_speaker'],
                        'anchor_utterances': anchor['utterances'],
                        'anchor_conversation': anchor['conversation'],
                        'paired_dataset': sample['dataset'],
                        'paired_speaker': sample['target_speaker'],
                        'paired_utterances': sample['utterances'],
                        'paired_conversation': sample['conversation'],
                        'label': label,
                    }, ensure_ascii=False
                ) + '\n'
            )

    with open(f'../output/unseen_user_base_{fold}.jsonl', 'w') as f:
        for triple in unseen_user_pairs[fold]:
            anchor, sample, label = triple
            f.write(
                json.dumps(
                    {
                        'anchor_dataset': anchor['dataset'],
                        'anchor_speaker': anchor['target_speaker'],
                        'anchor_utterances': anchor['utterances'],
                        'anchor_conversation': anchor['conversation'],
                        'paired_dataset': sample['dataset'],
                        'paired_speaker': sample['target_speaker'],
                        'paired_utterances': sample['utterances'],
                        'paired_conversation': sample['conversation'],
                        'label': label,
                    }, ensure_ascii=False
                ) + '\n'
            )

    with open(f'../output/seen_test_hard_{fold}.jsonl', 'w') as f:
        for triple in seen_conv_hard_pairs[fold]:
            anchor, sample, label = triple
            f.write(
                json.dumps(
                    {
                        'anchor_dataset': anchor['dataset'],
                        'anchor_speaker': anchor['target_speaker'],
                        'anchor_utterances': anchor['utterances'],
                        'anchor_conversation': anchor['conversation'],
                        'paired_dataset': sample['dataset'],
                        'paired_speaker': sample['target_speaker'],
                        'paired_utterances': sample['utterances'],
                        'paired_conversation': sample['conversation'],
                        'label': label,
                    }, ensure_ascii=False
                ) + '\n'
            )

    with open(f'../output/unseen_conv_hard_{fold}.jsonl', 'w') as f:
        for triple in unseen_conv_hard_pairs[fold]:
            anchor, sample, label = triple
            f.write(
                json.dumps(
                    {
                        'anchor_dataset': anchor['dataset'],
                        'anchor_speaker': anchor['target_speaker'],
                        'anchor_utterances': anchor['utterances'],
                        'anchor_conversation': anchor['conversation'],
                        'paired_dataset': sample['dataset'],
                        'paired_speaker': sample['target_speaker'],
                        'paired_utterances': sample['utterances'],
                        'paired_conversation': sample['conversation'],
                        'label': label,
                    }, ensure_ascii=False
                ) + '\n'
            )

    with open(f'../output/unseen_user_hard_{fold}.jsonl', 'w') as f:
        for triple in unseen_user_hard_pairs[fold]:
            anchor, sample, label = triple
            f.write(
                json.dumps(
                    {
                        'anchor_dataset': anchor['dataset'],
                        'anchor_speaker': anchor['target_speaker'],
                        'anchor_utterances': anchor['utterances'],
                        'anchor_conversation': anchor['conversation'],
                        'paired_dataset': sample['dataset'],
                        'paired_speaker': sample['target_speaker'],
                        'paired_utterances': sample['utterances'],
                        'paired_conversation': sample['conversation'],
                        'label': label,
                    }, ensure_ascii=False
                ) + '\n'
            )

    with open(f'../output/seen_test_harder_{fold}.jsonl', 'w') as f:
        for triple in seen_conv_harder_pairs[fold]:
            anchor, sample, label = triple
            f.write(
                json.dumps(
                    {
                        'anchor_dataset': anchor['dataset'],
                        'anchor_speaker': anchor['target_speaker'],
                        'anchor_utterances': anchor['utterances'],
                        'anchor_conversation': anchor['conversation'],
                        'paired_dataset': sample['dataset'],
                        'paired_speaker': sample['target_speaker'],
                        'paired_utterances': sample['utterances'],
                        'paired_conversation': sample['conversation'],
                        'label': label,
                    }, ensure_ascii=False
                ) + '\n'
            )

    with open(f'../output/unseen_conv_harder_{fold}.jsonl', 'w') as f:
        for triple in unseen_conv_harder_pairs[fold]:
            anchor, sample, label = triple
            f.write(
                json.dumps(
                    {
                        'anchor_dataset': anchor['dataset'],
                        'anchor_speaker': anchor['target_speaker'],
                        'anchor_utterances': anchor['utterances'],
                        'anchor_conversation': anchor['conversation'],
                        'paired_dataset': sample['dataset'],
                        'paired_speaker': sample['target_speaker'],
                        'paired_utterances': sample['utterances'],
                        'paired_conversation': sample['conversation'],
                        'label': label,
                    }, ensure_ascii=False
                ) + '\n'
            )

    with open(f'../output/unseen_user_harder_{fold}.jsonl', 'w') as f:
        for triple in unseen_user_harder_pairs[fold]:
            anchor, sample, label = triple
            f.write(
                json.dumps(
                    {
                        'anchor_dataset': anchor['dataset'],
                        'anchor_speaker': anchor['target_speaker'],
                        'anchor_utterances': anchor['utterances'],
                        'anchor_conversation': anchor['conversation'],
                        'paired_dataset': sample['dataset'],
                        'paired_speaker': sample['target_speaker'],
                        'paired_utterances': sample['utterances'],
                        'paired_conversation': sample['conversation'],
                        'label': label,
                    }, ensure_ascii=False
                ) + '\n'
            )
