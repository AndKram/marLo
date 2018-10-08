import time
import json
import gym
import numpy as np
import marlo
import marlo.xml
import malmoenv
import malmoenv.commands
import uuid
import hashlib
import base64
from lxml import etree

import traceback

from jinja2 import Environment as jinja2Environment
from jinja2 import FileSystemLoader as jinja2FileSystemLoader

import logging
logger = logging.getLogger(__name__)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class MarloEnvBuilderBase(gym.Env):
    """Base class for all Marlo environment builders
        
    All the individual ``MarloEnvBuilder`` classes 
    (for example: :class:`marlo.envs.DefaultWorld.main.MarloEnvBuilder`) 
    derive from this class.
    This class provides all the necessary functions for the 
    lifecycle management of a MarLo environment.
    """     
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, templates_folder):
        self.jinja2_fileloader = None
        self.jinj2_env = None
        super(MarloEnvBuilderBase, self).__init__()

        self.templates_folder = templates_folder
        self.setup_templating()
        self._default_base_params = False
        self.mission_spec = None
        self.experiment_id = None
        self.env = None
        self._rounds = 0
        self.dry_run = False
        self.video_height = 600
        self.video_width = 800
        self.video_depth = 3
        self.observation_space = None
        self.last_image = None
        self.action_names = []
        self.action_spaces = []
        self.action_space = None
        self.port = 9000
        self.server = '127.0.0.1'
        self.step_sleep = 0

    def setup_templating(self):
        """
            Sets up the basic ``jinja2`` templating fileloader and 
            environments.
            The ``MarloEnvBuilder`` classes, expect the following variables 
            to be available to them for rendering the ``MissionSpec``
            
            - ``self.jinja2_fileloader``
            - ``self.jinj2_env``
        """
        self.jinja2_fileloader = jinja2FileSystemLoader(self.templates_folder)
        self.jinj2_env = jinja2Environment(loader=self.jinja2_fileloader)

    def render_mission_spec(self):
        """
            This function looks for a ``mission.xml`` template inside the 
            ``templates`` folder, and renders it using ``jinja2``.
            This can very well be overriden by ``MarloEnvBuilder`` if required.
        """
        template = self.jinj2_env.get_template("mission.xml")
        return template.render(
            params=self.params
        )

    @property
    def white_listed_join_params(self):
        """
            This returns a list of whitelisted game parameters which can be
            modified when joining a game by using :meth:`marlo.init`.
        """
        return marlo.JOIN_WHITELISTED_PARAMS

    @property
    def default_base_params(self):
        """
            The **default game parameters** for all MarLo environments. 
            These can be modified by either overriding this class in 
            :class:`marlo.envs.DefaultWorld.main.MarloEnvBuilder` or implementing 
            a `_default_params` function in the derived class.
            
            The default parameters are as follows :

            :param seed: Seed for the random number generator (Default : ``random``). (**Note** This is not properly integrated yet.)
            :type seed: int
            
            :param tick_length: length of a single in-game tick (in milliseconds) (Default : ``50``)
            :type tick_length: int 
            
            :param role: Game Role with which the current agent should join. (Default : ``0``)
            :type role: int
            
            :param experiment_id: A unique alphanumeric id for a single game. This is used to validate the session that an agent is joining. (Default : ``random_experiment_id``). 
            :type experiment_id: str
            
            :param port: The base port for Minecraft connections. Role 0 uses this port and Role N uses port + N.
            :type port: int

            :param server: The server for Minecraft connections.
            :type server: str

            :param agent_names: A `list` of names for the agents that are expected to join the game. This is used by the templating system to add an appropriate number of agents. (Default : ``["MarLo-Agent-0"]``)
            :type client_pool: list
            
            :param max_retries: Maximum Number of retries when trying to connect to a client_pool to start a mission. (Default : ``30``)
            :type max_retries: int
            
            :param retry_sleep: Time (in seconds) that the execution should sleep between retries for starting a mission. (Default: ``3``)
            :type retry_sleep: float
            
            :param step_sleep: Time (in seconds) to sleep when trying to obtain the latest world state. (Default: ``0.001``)
            :type step_sleep: float
            
            :param skip_steps: Number of observation steps to skip everytime we attempt to the latest world_state. (Default: ``0``) 
            :type skip_steps: int
            
            :param videoResolution: Resolution of the frame that is expected as the RGB observation. (Default: ``[800, 600]``)
            :type videoResolution: list
            
            :param videoWithDepth: If the depth channel should also be added to the observation. (Default: ``False`` )
            :type videoWithDepth: bool
            
            :param prioritise_offscreen_rendering: Prioritise Off Screen Rendering. Can be useful for better render speeds during training. And should be set as False when debugging. (Default: True)
            :type prioritise_offscreen_rendering: bool
            
            :param observeRecentCommands: If the Recent Commands should be included in the auxillary observation available through ``info['observation']``. (Default: ``False``)
            :type observeRecentCommands: bool

            :param observeHotBar: If the HotBar information should be included in the auxillary observation available through ``info['observation']``. (Default: ``False``)
            :type observeHotBar: bool
            
            :param observeFullInventory: If the FullInventory information should be included in the auxillary observation available through ``info['observation']``. (Default: ``False``)
            :type observeFullInventory: bool

            :param observeGrid: Asks for observations of the block types within a cuboid relative to the agent's position in the auxillary observation available through ``info['observation']``. (Default: ``False``)
            :type observeGrid: bool, list
            
            :param observeDistance: Asks for the Euclidean distance to a location to be included in the auxillary observation available through ``info['observation']``. (Default: ``False``)
            :type observeDistance: bool, list

            :param observeChat: If the Chat information should be included in the auxillary observation available through ``info['observation']``. (Default: ``False``)
            :type observeChat: bool
            
            TODO :param continuous_to_discrete: Converts continuous actions to discrete. when allowed continuous actions are 'move' and 'turn', then discrete action space contains 4 actions: move -1, move 1, turn -1, turn 1. (Default : ``True``)
            TODO :type continuous_to_discrete: bool
            
            TODO :param allowContinuousMovement: If all continuous movement commands should be allowed. (Default : ``False``)
            TODO :type allowContinuousMovement: bool
            
            TODO :param allowDiscreteMovement: If all discrete movement commands should be allowed. (Default : ``False``)
            TODO :type allowDiscreteMovement: bool
            
            TODO :param allowAbsoluteMovement: If all absolute movement commands should be allowed. (Default : ``False``) (**Not Implemented**)
            TODO :type allowAbsoluteMovement: bool
            
            :param add_noop_command: If a ``noop`` (``move 0\\nturn 0``) command should be added to the actions. (Default : ``False``)
            :type add_noop_command: bool
            TODO DOES NOT WORK WITH TURN BASED COMMANDS
            
            TODO :param recordDestination: Destination where Mission Records should be stored. (Default : ``None``)
            TODO :type recordDestination: str
            
            TODO :param recordObservations: If Observations should be recorded in the ``MissionRecord``s. (Default : ``None``)
            TODO :type recordObservations: bool
            
            TODO :param recordRewards: If Rewards should be recorded in the ``MissionRecord``s. (Default : ``None``)
            TODO :type recordRewards: bool
            
            TODO :param recordCommands: If Commands (actions) should be recorded in the ``MissionRecord``s. (Default : ``None``)
            TODO :type recordCommands: bool

            TODO :param recordMP4: If a MP4 should be recorded in the ``MissionRecord``, and if so, the specifications as : ``[frame_rate, bit_rate]``.  (Default : ``None``)
            TODO :type recordMP4: list

            :param gameMode: The Minecraft gameMode for this particular game. One of ``['spectator', 'creative', 'survival']``. (Default: ``survival``)
            :type gameMode: str

            TODO :param forceWorldReset: Force world reset on every reset. Makes sense only in case of environments with inherent stochasticity (Default: ``False``)
            TODO :type forceWorldReset: bool

            :param comp_all_commands: Specifies the superset of allowed commands in Marlo competition. (Default : ``['move', "turn", "use", "attack"]``)
            :type comp_all_commands: list of strings
        """
        if not self._default_base_params:
            self._default_base_params = dotdict(
                 seed="random",
                 tick_length=50,
                 role=0,
                 experiment_id="random_experiment_id",
                 agent_names=["MarLo-Agent-0"],
                 port=9000,
                 server='127.0.0.1',
                 max_retries=30,
                 retry_sleep=3,
                 step_sleep=0.001,
                 skip_steps=0,
                 videoResolution=[800, 600],
                 videoWithDepth=None,
                 prioritise_offscreen_rendering=True,
                 observeRecentCommands=None,
                 observeHotBar=None,
                 observeFullInventory=None,
                 observeGrid=None,
                 observeDistance=None,
                 observeChat=None,
                 continuous_to_discrete=True,
                 allowContinuousMovement=False,
                 allowDiscreteMovement=False,
                 allowAbsoluteMovement=False,
                 add_noop_command=False,
                 recordDestination=None,
                 recordObservations=None,
                 recordRewards=None,
                 recordCommands=None,
                 recordMP4=None,
                 gameMode="survival",
                 forceWorldReset=False,
                 comp_all_commands=['move', "turn", "use", "attack"]
            )
        return self._default_base_params

    def setup_video(self, params):
        """
            Setups up the Video Requests for an environment.
            
            :param params: Marlo Game Parameters as described in :meth:`default_base_params`
            :type params: dict
        """
        ############################################################
        # Setup Video
        ############################################################
        vid = 'Mission.AgentSection.AgentHandlers.VideoProducer'
        if params.videoResolution:
            marlo.xml.put(self.mission_spec, vid + '.Width', params.videoResolution[0])
            marlo.xml.put(self.mission_spec, vid + '.Height', params.videoResolution[1])
        if params.videoWithDepth:
            marlo.xml.put(self.mission_spec, vid, "true", attrib="want_depth")

    def setup_observe_params(self, params):
        """
            Setups up the Auxillary Observation Requests for an environment.
            
            :param params: Marlo Game Parameters as described in :meth:`default_base_params`
            :type params: dict
        """
        ############################################################
        # Setup observe<>*
        ############################################################
        if params.observeRecentCommands:
            marlo.xml.put(self.mission_spec, "Mission.AgentSection.AgentHandlers.ObservationFromRecentCommands", '')
        if params.observeHotBar:
            marlo.xml.put(self.mission_spec, "Mission.AgentSection.AgentHandlers.ObservationFromHotBar", '')
        if params.observeFullInventory:
            marlo.xml.put(self.mission_spec, "Mission.AgentSection.AgentHandlers.ObservationFromFullInventory", '')
        if params.observeGrid:
            self._observe_grid(*(params.observeGrid + ["grid"]))
        if params.observeDistance:
            self._observe_distance(
                 *(params.observeDistance + ["dist"])
                 )
        if params.observeChat:
            marlo.xml.put(self.mission_spec, "Mission.AgentSection.AgentHandlers.ObservationFromChat", '')

    def _observe_grid(self, x1, y1, z1, x2, u2, z2, name):
        tag = "Mission.AgentSection.AgentHandlers.ObservationFromGrid.Grid"
        marlo.xml.put(self.mission_spec, tag, '')
        e = marlo.xml.get_sub_element(self.mission_spec, tag)
        e.arrtib["name"] = name
        tag = "Mission.AgentSection.AgentHandlers.ObservationFromGrid.Grid.min"
        marlo.xml.put(self.mission_spec, tag, '')
        e = marlo.xml.get_sub_element(self.mission_spec, tag)
        e.arrtib["x"] = x1
        e.arrtib["y"] = y1
        e.arrtib["z"] = z1
        tag = "Mission.AgentSection.AgentHandlers.ObservationFromGrid.Grid.max"
        marlo.xml.put(self.mission_spec, tag, '')
        e = marlo.xml.get_sub_element(self.mission_spec, tag)
        e.arrtib["x"] = x2
        e.arrtib["y"] = y2
        e.arrtib["z"] = z2

    def _observe_distance(self, x, y, z, name):
        tag = "Mission.AgentSection.AgentHandlers.ObservationFromDistance.Marker"
        marlo.xml.put(self.mission_spec, tag, '')
        e = marlo.xml.get_sub_element(self.mission_spec, tag)
        e.arrtib["name"] = name
        e.arrtib["x"] = x
        e.arrtib["y"] = y
        e.arrtib["z"] = z

    def setup_observation_space(self, params):
        """
            Setups up the Observation Space for an environment.
            
            :param params: Marlo Game Parameters as described in :meth:`default_base_params`
            :type params: dict
        """        
        ############################################################
        # Setup Observation Space
        ############################################################
        self.video_height = params.videoResolution[1]
        self.video_width = params.videoResolution[0]
        self.video_depth = 4 if params.videoWithDepth else 3
        self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=(self.video_height, self.video_width, self.video_depth),
                dtype=np.uint8
                )
        # Setup a dummy first image
        self.last_image = np.zeros(
            (self.video_height, self.video_width, self.video_depth),
            dtype=np.uint8
            )

    def setup_action_space(self, params):
        """
            Setups up the action space for the current agent interacting with the environment.
            
            :param params: Marlo Game Parameters as described in :meth:`default_base_params`
            :type params: dict
        """
        ############################################################
        # Setup Action Space
        ############################################################
        continuous_actions = []
        discrete_actions = []
        multidiscrete_actions = []
        multidiscrete_action_ranges = []
        if params.add_noop_command:
            discrete_actions.append("move 0\nturn 0")  # Does not work with turn key.

        mission_xml = etree.tostring(self.mission_spec).decode()
        i = mission_xml.index("<Mission")
        mission_xml = mission_xml[i:]
        # print(mission_xml)
        parser = malmoenv.commands.CommandParser(params.comp_all_commands)
        commands = parser.get_commands(mission_xml, params.role)

        for (command_handler, turnbased, command) in commands:
            logger.debug("CommandHandler: {} turn based: {} command: {} ".format(command_handler, turnbased, command))

            if command_handler == "ContinuousMovement":
                if command in ["move", "strafe", "pitch", "turn"]:
                    if params.continuous_to_discrete:
                        discrete_actions.append(command + " 1")
                        discrete_actions.append(command + " -1")
                    else:
                        continuous_actions.append(command)
                elif command in ["crouch", "jump", "attack", "use"]:
                    if params.continuous_to_discrete:
                        discrete_actions.append(command + " 1")
                        discrete_actions.append(command + " 0")
                    else:
                        multidiscrete_actions.append(command)
                        multidiscrete_action_ranges.append([0, 1])
                else:
                    raise ValueError(
                        "Unknown continuous action : {}".format(command)
                    )
            elif command_handler == "DiscreteMovement":
                if command in marlo.SINGLE_DIRECTION_DISCRETE_MOVEMENTS:
                    discrete_actions.append(command + " 1")
                elif command in marlo.MULTIPLE_DIRECTION_DISCRETE_MOVEMENTS:
                    discrete_actions.append(command + " 1")
                    discrete_actions.append(command + " -1")
                else:
                    raise ValueError(
                        "Unknown discrete action : {}".format(command)
                    )
            elif command_handler in ["AbsoluteMovement", "Inventory"]:
                logger.warn(
                    "Command Handler `{}` Not Implemented".format(
                        command_handler
                    )
                )
            elif command_handler in ["MissionQuit"]:
                logger.debug(
                    "Command Handler `{}`".format(
                        command_handler
                    )
                )
            else:
                raise ValueError(
                    "Unknown Command Handler : `{}`".format(
                        command_handler
                    )
                )
        # Convert lists into proper gym action spaces
        self.action_names = []
        self.action_spaces = []

        # Discrete Actions
        if len(discrete_actions) > 0:
            self.action_spaces.append(
                gym.spaces.Discrete(len(discrete_actions))
                )
            self.action_names.append(discrete_actions)
        # Continuous Actions
        if len(continuous_actions) > 0:
            self.action_spaces.append(
                gym.spaces.Box(-1, 1, (len(continuous_actions),))
                )
            self.action_names.append(continuous_actions)
        if len(multidiscrete_actions) > 0:
            self.action_spaces.append(
                gym.spaces.MultiDiscrete(multidiscrete_action_ranges)
                )
            self.action_names.append(multidiscrete_actions)

        # No tuples in case a single action
        if len(self.action_spaces) == 1:
            self.action_space = self.action_spaces[0]
        else:
            self.action_space = gym.spaces.Tuple(self.action_space)

    # def setup_action_commands(self, params):
    #    """
    #       Setups up the Action Commands for the current agent interacting with the environment.
    #
    #       :param params: Marlo Game Parameters as described in :meth:`default_base_params`
    #        :type params: dict
    #   """
        ############################################################
        # Setup Action Commands
        ############################################################
        # if params.allowContinuousMovement or params.allowAbsoluteMovement or \
        #         params.allowDiscreteMovement:
        #     # Remove all command handlers
        #     # self.mission_spec.removeAllCommandHandlers()
        #
        #     # ContinousMovement commands
        #     if isinstance(params.allowContinuousMovement, list):
        #         for _command in params.allowContinuousMovement:
        #             self.mission_spec.allowContinuousMovementCommand(_command)
        #     elif params.allowContinuousMovement is True:
        #         self.mission_spec.allowAllContinuousMovementCommands()
        #
        #     # AbsoluteMovement commands
        #     if isinstance(params.allowAbsoluteMovement, list):
        #         for _command in params.allowAbsoluteMovement:
        #             self.mission_spec.allowAbsoluteMovementCommand(_command)
        #     elif params.allowAbsoluteMovement is True:
        #         self.mission_spec.allowAllAbsoluteMovementCommands()
        #
        #     # DiscreteMovement commands
        #     if isinstance(params.allowDiscreteMovement, list):
        #         for _command in params.allowDiscreteMovement:
        #             self.mission_spec.allowDiscreteMovementCommand(_command)
        #     elif params.allowDiscreteMovement is True:
        #         self.mission_spec.allowAllDiscreteMovementCommands()

    # def setup_mission_record(self, params):
    #    """
    #       Setups up the ``mission_record`` for the current environment.
    #
    #       :param params: Marlo Game Parameters as described in :meth:`default_base_params`
    #       :type params: dict
    #   """
        ############################################################
        # Setup Mission Record
        ############################################################
        # self.mission_record_spec = MalmoPython.MissionRecordSpec() # empty
        # if params.recordDestination:
        #     if not params.recordDestination.endswith(".tgz"):
        #         raise Exception("Invalid recordDestination provided"
        #                         "recordDestination should be a valid path ending"
        #                         " with .tgz ")
        #     self.mission_record_spec.setDestination(params.recordDestination)
        #     if params.recordRewards:
        #         self.mission_record_spec.recordRewards()
        #     if params.recordCommands:
        #         self.mission_record_spec.recordCommands()
        #     if params.recordMP4:
        #         assert type(params.recordMP4) == list \
        #             and len(params.recordMP4) == 2
        #         self.mission_record_spec.recordMP4(*(params.recordMP4))
        # else:
        #     if params.recordRewards or params.recordCommands  or params.recordMP4:
        #         raise Exception("recordRewards or recordCommands or recordMP4 "
        #                         "provided without specifyin recordDestination")

    def setup_game_mode(self, params):
        """
            Setups up the ``gameMode`` for the current environment.
            
            :param params: Marlo Game Parameters as described in :meth:`default_base_params`
            :type params: dict
        """                
        ############################################################
        # Setup Game Mode
        ############################################################
        if params.gameMode:
            if params.gameMode == "spectator":
                marlo.xml.put(self.mission_spec, "Mission.AgentSection", "Spectator", attrib="mode")
            elif params.gameMode == "creative":
                marlo.xml.put(self.mission_spec, "Mission.AgentSection", "Creative", attrib="mode")
            elif params.gameMode == "survival":
                logger.info("params.gameMode : Cannot force survival mode.")
            else:
                raise Exception("Unknown params.gameMode : {}".format(
                    params.gameMode
                ))

    def setup_mission_spec(self):
        """
            Generates and setups the first MissionSpec as generated by :meth:`render_mission_spec`.
        """                
        ############################################################
        # Instantiate Mission Spec
        ############################################################
        mission_xml = self.render_mission_spec()
        if not mission_xml.startswith('<Mission'):
            i = mission_xml.index("<Mission")
            mission_xml = mission_xml[i:]
        self.mission_spec = etree.fromstring(mission_xml)

    def init(self, params, dry_run=False):
        """
            Generates the join tokens for all the agents in a game based on the provided game params.
            
            :param params: Marlo Game Parameters as described in :meth:`default_base_params`
            :type params: dict
            :param dry_run: If the current execution is a ``dry_run``
            :type dry_run: bool
            
            :returns: List of join_tokens, one join_token for every agent in the game.
            :rtype: list
        """                                
        self.params.update(params)
        self.port = self.params["port"]
        self.server = self.params["server"]
        self.step_sleep = self.params["step_sleep"]
        self.dry_run = dry_run
        self.build_env(self.params)
        mission_xml = etree.tostring(self.mission_spec).decode()
        role = params.get("role", 0)
        print("init role " + str(role))
        experiment_id = params.get("experiment_id", None)
        if not experiment_id:
            experiment_id = str(uuid.uuid4())
        self.env.init(mission_xml, self.port, role=role, port2=(self.port + role), server=self.server,
                      action_space=malmoenv.StringActionSpace(),
                      exp_uid=experiment_id)
        number_of_agents = self.env.agent_count

        join_tokens = []
        for _idx in range(number_of_agents):
            _join_token = {
                'role': _idx,
                'mission_xml': mission_xml,
                'experiment_id': experiment_id,
                'game_params': self.params
            }
            _join_token = base64.b64encode(
                    json.dumps(_join_token).encode('utf8')
            )
            join_tokens.append(_join_token)
        return join_tokens

    def build_env(self, params):
        self.setup_mission_spec()

        self.setup_video(params)
        self.setup_observe_params(params)
        # self.setup_action_commands(params) TODO Don't allow override - let the xml dictate the action space.
        self.setup_observation_space(params)
        self.setup_action_space(params)

        # self.setup_mission_record(params) TODO malmoenv's test.py can records every Nth frame.
        self.setup_game_mode(params)
        self.env = malmoenv.make()

    ########################################################################
    # Env interaction functions
    ########################################################################
        
    def reset(self):

        image = self.env.reset()
        self.last_image = image

        # Notify Evaluation System, if applicable
        marlo.CrowdAiNotifier._env_reset()

        return image

    def _get_action_string(self, actions):
        # no tuple in case of a single action
        if len(self.action_spaces) == 1:
            actions = [actions]

        # send corresponding command
        for _spaces, _commands, _actions in \
                zip(self.action_spaces, self.action_names, actions):

            if isinstance(_spaces, gym.spaces.Discrete):
                logger.debug(_commands[_actions])
                return _commands[_actions]
            elif isinstance(_spaces, gym.spaces.Box):
                for command, value in zip(_commands, _actions):
                    _command = "{}-{}".format(command, value)
                    logger.debug(_command)
                    return _command
            elif isinstance(_spaces, gym.spaces.MultiDiscrete):
                for command, value in zip(_commands, _actions):
                    _command = "{}-{}".format(command, value)
                    logger.debug(_command)
                    return _command
            else:
                logger.warn("Ignoring unknown action space for {}".format(
                    _commands
                ))

    def _step_wrapper(self, action):
        cmd = self._get_action_string(action)
        # print("[" + str(self.env.role) + "] cmd [" + cmd + "]")
        image, reward, done, info = self.env.step(cmd)
        self.last_image = image
        # Notify evaluation system, if applicable
        # marlo.CrowdAiNotifier._env_action(action)
        marlo.CrowdAiNotifier._step_reward(reward)
        if done:
            marlo.CrowdAiNotifier._episode_done()

        time.sleep(self.step_sleep)  # yield
        return image, reward, done, info

    def step(self, action):
        """
            Helps wrap the actual step function to catch relevant errors
        """
        try:
            return self._step_wrapper(action)
        except Exception as e:
            marlo.CrowdAiNotifier._env_error(str(e))
            raise e

    def render(self, mode='rgb_array', close=False):
        if mode == "rgb_array":
            return self.last_image
        elif mode == "human":
            # TODO: Implement this
            raise None
        else:
            raise NotImplemented("Render Mode not implemented : {}"
                                 .format(mode))

    def seed(self, seed=None):
        if seed is None:
            return
        e = marlo.xml.get_sub_element(self.mission_spec, "Mission.ServerSection.ServerHandlers.DefaultWorldGenerator")
        if e is not None:
            e.attrib["seed"] = str(seed)
        e = marlo.xml.get_sub_element(self.mission_spec, "Mission.ServerSection.ServerHandlers.FlatWorldGenerator")
        if e is not None:
            e.attrib["seed"] = str(seed)
        return [seed]

