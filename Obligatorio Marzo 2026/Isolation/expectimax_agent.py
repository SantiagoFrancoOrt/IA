import math
from collections import deque

from agent import Agent
from board import Board


class ExpectimaxAgent(Agent):
    """
    Expectimax: el agente maximiza su utilidad mientras modela al oponente
    como un jugador estocástico que elige acciones uniformemente (nodo CHANCE).
    Esto lo hace más robusto contra oponentes subóptimos comparado con Minimax.
    Basado en la estructura del práctico 7 (AgentExpectimax sobre TicTacToe).
    """

    def __init__(self, player=1, depth=3):
        super().__init__(player)
        self.depth = depth

    # ------------------------------------------------------------------
    # Interface requerida por Agent
    # ------------------------------------------------------------------

    def next_action(self, obs: Board):
        action, _ = self.expectimax(obs, self.player, self.depth)
        return action

    def heuristic_utility(self, board: Board) -> float:
        """Misma evaluación que MinimaxAgent para comparación justa."""
        reach_me = self._reachable_cells(board, self.player)
        reach_op = self._reachable_cells(board, 3 - self.player)
        mob_me = len(board.get_possible_actions(self.player))
        mob_op = len(board.get_possible_actions(3 - self.player))
        cent = self._centrality(board, self.player) - self._centrality(board, 3 - self.player)
        return 2.0 * (reach_me - reach_op) + 1.0 * (mob_me - mob_op) + 0.5 * cent

    # ------------------------------------------------------------------
    # Funciones de evaluación individuales
    # ------------------------------------------------------------------

    def _reachable_cells(self, board: Board, player: int) -> int:
        """BFS desde la posición del jugador; cuenta celdas vacías alcanzables."""
        start = board.find_player_position(player)
        if start is None:
            return 0
        visited = {start}
        queue = deque([start])
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
        while queue:
            r, c = queue.popleft()
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < board.board_size[0]
                    and 0 <= nc < board.board_size[1]
                    and (nr, nc) not in visited
                    and board.grid[nr, nc] == 0
                ):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return len(visited) - 1

    def _centrality(self, board: Board, player: int) -> float:
        """Negativo de la distancia Manhattan desde el centro del tablero."""
        pos = board.find_player_position(player)
        if pos is None:
            return 0.0
        cr = (board.board_size[0] - 1) / 2
        cc = (board.board_size[1] - 1) / 2
        return -(abs(pos[0] - cr) + abs(pos[1] - cc))

    # ------------------------------------------------------------------
    # Expectimax
    # ------------------------------------------------------------------

    def expectimax(self, board: Board, current_player: int, depth: int):
        # Caso base: fin del juego
        done, winner = board.is_end(current_player)
        if done:
            score = math.inf if winner == self.player else -math.inf
            return None, score

        # Caso base: profundidad agotada
        if depth == 0:
            return None, self.heuristic_utility(board)

        actions = board.get_possible_actions(current_player)
        best_action = actions[0] if actions else None

        if current_player == self.player:  # nodo MAX
            best_val = -math.inf
            for action in actions:
                child_node = board.clone()
                child_node.play(action, current_player)
                _, val = self.expectimax(child_node, 3 - current_player, depth - 1)
                if val > best_val:
                    best_val = val
                    best_action = action
            return best_action, best_val

        else:  # nodo CHANCE — promedio uniforme sobre acciones del oponente
            total = 0.0
            for action in actions:
                child_node = board.clone()
                child_node.play(action, current_player)
                _, val = self.expectimax(child_node, 3 - current_player, depth - 1)
                total += val
            expected = total / len(actions)
            return None, expected