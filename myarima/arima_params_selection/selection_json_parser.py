import json


def get_max_orders(file):
    p = []
    d = []
    q = []
    P = []
    D = []
    Q = []
    with open(file) as json_file:
        data = json.load(json_file)
        for obj in data:
            for inner_obj in obj['inner']:
                order = inner_obj['order']
                seasonal_order = inner_obj['seasonal_order']
                p.append(order[0])
                d.append(order[1])
                q.append(order[2])
                P.append(seasonal_order[0])
                D.append(seasonal_order[1])
                Q.append(seasonal_order[2])
    print('Min p: %d, Max p: %d' % (min(p), max(p)))
    print('Min d: %d, Max d: %d' % (min(d), max(d)))
    print('Min q: %d, Max q: %d' % (min(q), max(q)))

    print('Min P: %d, Max P: %d' % (min(P), max(P)))
    print('Min D: %d, Max D: %d' % (min(D), max(D)))
    print('Min Q: %d, Max Q: %d' % (min(Q), max(Q)))

