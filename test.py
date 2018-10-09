import marlo

from threading import Thread
import threading
from lxml import etree
import argparse
import traceback
import sys
import time
from PIL import Image


def main():
    xml_file = "mission.xml"
    parser = argparse.ArgumentParser(description='Multi-agent test')
    parser.add_argument('--rounds', type=int, default=10, help='number of rollouts - default 10')
    parser.add_argument('--mission', type=str, default=xml_file, help='the env or mission xml')
    parser.add_argument('--agent_count', type=int, default=None, help='number of agents')
    parser.add_argument('--server', type=str, default="127.0.0.1", help='the mission server')
    parser.add_argument('--port', type=int, default=10000, help='the mission port')
    parser.add_argument('--saveimagesteps', type=int, default=0, help='save an image every N steps')
    args = parser.parse_args()

    rounds = args.rounds

    if args.agent_count is None:
        xml_file = args.mission
        xml = etree.parse(xml_file)
        number_of_agents = len(xml.getroot().findall('{http://ProjectMalmo.microsoft.com}AgentSection'))
    else:
        number_of_agents = args.agent_count

    join_tokens = marlo.make(args.mission,
                             params=dict(server=args.server, port=args.port, turn_based=True,
                                         comp_all_commands=['move', "turn", "use", "attack"]
                                         ))

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

        steps = 0
        for r in range(rounds):
            print("reset agent " + str(role) + " for new game " + str(r + 1))
            obs = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                steps += 1
                # print("obs:", obs)
                # print("reward:", reward)
                # print("done:", done)
                # print("info", info)
                if args.saveimagesteps > 0 and steps % args.saveimagesteps == 0:
                    img = Image.fromarray(obs)
                    img.save('image' + str(steps) + '.png')
        env.close()

    threads = [Thread(target=run, args=(join_tokens[i], i, rounds)) for i in range(number_of_agents)]

    [t.start() for t in threads]
    if False:
        thread_dumper = Thread(target=dump)
        thread_dumper.start()
    [t.join() for t in threads]


if __name__ == "__main__":
    main()
