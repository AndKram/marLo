import marlo

from threading import Thread
import threading
from lxml import etree
import argparse
import traceback
import sys
import time


def main():
    xml_file = "mission.xml"
    parser = argparse.ArgumentParser(description='Multi-agent test')
    parser.add_argument('--rounds', type=int, default=10, help='number of rollouts - default 10')
    parser.add_argument('--mission_file', type=str, default=xml_file, help='the mission xml')
    args = parser.parse_args()

    rounds = args.rounds
    xml_file = args.mission_file

    xml = etree.parse(xml_file)
    number_of_agents = len(xml.getroot().findall('{http://ProjectMalmo.microsoft.com}AgentSection'))

    client_pool = [('127.0.0.1', 10000 + i) for i in range(number_of_agents)]

    join_tokens = marlo.make(xml_file, params=dict(client_pool=client_pool, turn_based=True,
                                                   comp_all_commands=['move', "turn", "use", "attack"],
                                                   kill_clients_after_num_rounds=250))

    def dump():
        while True:
            for th in threading.enumerate():
                print(th)
                traceback.print_stack(sys._current_frames()[th.ident])
                print()
            time.sleep(30)

    def run(join_token, role, rounds):
        print("agent " + str(role))

        env = marlo.init(join_token)
        env.seed(4711)

        for r in range(rounds):
            print("reset agent " + str(role) + " for new game " + str(r + 1))
            obs = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                # print("obs:", obs)
                # print("reward:", reward)
                # print("done:", done)
                # print("info", info)
        env.close()

    threads = [Thread(target=run, args=(join_tokens[i], i, rounds)) for i in range(number_of_agents)]

    [t.start() for t in threads]
    if False:
        thread_dumper = Thread(target=dump)
        thread_dumper.start()
    [t.join() for t in threads]


if __name__ == "__main__":
    main()
