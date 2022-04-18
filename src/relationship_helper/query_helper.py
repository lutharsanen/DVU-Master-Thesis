import networkx as nx


def get_source(G, target, relation):
    sources = []
    if target != "blank":
        if target not in G.nodes():
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
        for edges in G.edges.data():
            if edges[-1]["label"] == "Ex Partner Of":
                sources.append(i[0])
        
        return sources


def count_relations(G, subject, relation, target):
    if subject not in G.nodes():
        return "unknown"
    else:
        path = nx.single_target_shortest_path(G, target,1)
        counter = 0
        for lst in path.values():
            if len(lst) > 1:
                edge = G.get_edge_data(lst[0],lst[1])["label"]
                if edge == relation:
                    counter += 1
        return counter


def get_relations(G, subject, target):
    if target or subject not in G.nodes:
        return "unknown"
    else:
        paths = list(nx.all_simple_paths(G, source = subject, target=target))
        if len(paths) > 0:
            relation = G.get_edge_data(subject, target)
            return relation