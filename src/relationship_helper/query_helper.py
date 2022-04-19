import networkx as nx

def get_source(G, target, relation):
    sources = []
    if target != "blank":
        if target not in list(G.nodes()):
            return 0
        else:
            path = nx.single_target_shortest_path(G, target,1)
            for lst in path.values():
                if len(lst) > 1:
                    edge = G.get_edge_data(lst[0],lst[1])["label"]
                    if edge == relation:
                        sources.append(lst[0])

            return sources
    
    else:
        #print(G.edges.data())
        for edges in G.edges.data():
            if edges[-1]["label"] == relation:
                sources.append(edges[0])
        
        return sources
    
def count_relations(G, subject, relation):
    if subject not in list(G.nodes()):
        return "unknown"
    else:
        path = nx.single_target_shortest_path(G, subject,1)
        counter = 0
        for lst in path.values():
            if len(lst) > 1:
                edge = G.get_edge_data(lst[0],lst[1])["label"]
                if edge == relation:
                    counter += 1
        return counter

def get_relations(G, subject, target, answers, person_list, person_list_inverse, location_list, location_list_inverse):
    
    if (target not in list(G.nodes)) or (subject not in list(G.nodes)):
        return "unknown"
    else:
        paths = list(nx.all_simple_paths(G, source = subject, target=target, cutoff = 1))
        if len(paths) > 0:
            relation = G.get_edge_data(subject, target)
            proba = list(relation["weights"])
            probabilities = []
            tracked_answers = []
            for answer in answers:
                if answer in person_list:
                    tracked_answers.append(answer)
                    idx = person_list.index(answer)
                    if idx >= len(proba):
                        new_idx = proba.index(max(proba))
                        val = proba[new_idx]
                    else:
                        val = proba[idx]
                    probabilities.append(val)
                elif answer in person_list_inverse:
                    tracked_answers.append(answer)
                    idx = person_list_inverse.index(answer)
                    if idx >= len(proba):
                        new_idx = proba.index(max(proba))
                        val = proba[new_idx]
                    else:
                        val = proba[idx]
                    probabilities.append(val)
                elif answer in location_list:
                    tracked_answers.append(answer)
                    idx = location_list.index(answer)
                    if idx >= len(proba):
                        new_idx = proba.index(max(proba))
                        val = proba[new_idx]
                    else:
                        val = proba[idx]
                    probabilities.append(val)
                elif answer in location_list_inverse:
                    tracked_answers.append(answer)
                    idx = location_list_inverse.index(answer)
                    if idx >= len(proba):
                        new_idx = proba.index(max(proba))
                        val = proba[new_idx]
                    else:
                        val = proba[idx]
                    probabilities.append(val)
            max_idx = probabilities.index(max(probabilities))
            return answers[max_idx]