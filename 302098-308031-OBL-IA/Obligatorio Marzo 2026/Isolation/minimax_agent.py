import math
from collections import deque

from agent import Agent
from board import Board


class MinimaxAgent(Agent):
    """Minimax with Alpha-Beta Pruning."""

    def __init__(self, player=1, depth=3):
        super().__init__(player)
        self.depth = depth
        self._nodes_visited = 0  # for pruning analysis

    # ------------------------------------------------------------------
    # Interface requerida por Agent
    # ------------------------------------------------------------------

    def next_action(self, obs: Board):
        self._nodes_visited = 0
        action, _ = self._minimax(obs, self.player, self.depth, -math.inf, math.inf)
        return action

    def heuristic_utility(self, board: Board) -> float:
        """Combinación ponderada de tres funciones de evaluación."""
        reach_me = self._reachable_cells(board, self.player)
        reach_op = self._reachable_cells(board, 3 - self.player)
        mob_me = len(board.get_possible_actions(self.player))
        mob_op = len(board.get_possible_actions(3 - self.player))
        cent = self._centrality(board, self.player) - self._centrality(board, 3 - self.player)

        # Ponderación: reachability es la señal más fuerte en Isolation
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
        return len(visited) - 1  # no cuenta la posición de inicio

    def _centrality(self, board: Board, player: int) -> float:
        """Negativo de la distancia Manhattan desde el centro del tablero."""
        pos = board.find_player_position(player)
        if pos is None:
            return 0.0
        cr = (board.board_size[0] - 1) / 2
        cc = (board.board_size[1] - 1) / 2
        return -(abs(pos[0] - cr) + abs(pos[1] - cc))

    # ------------------------------------------------------------------
    # Minimax con Alpha-Beta Pruning
    # ------------------------------------------------------------------

    def _minimax(
        self,
        board: Board,
        current_player: int,
        depth: int,
        alpha: float,
        beta: float,
    ):
        self._nodes_visited += 1

        done, winner = board.is_end(current_player)
        if done:
            score = math.inf if winner == self.player else -math.inf
            return None, score

        if depth == 0:
            return None, self.heuristic_utility(board)

        actions = board.get_possible_actions(current_player)
        best_action = actions[0] if actions else None

        if current_player == self.player:  # nodo MAX
            best_val = -math.inf
            for action in actions:
                child = board.clone()
                child.play(action, current_player)
                _, val = self._minimax(child, 3 - current_player, depth - 1, alpha, beta)
                if val > best_val:
                    best_val = val
                    best_action = action
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break  # poda beta
            return best_action, best_val

        else:  # nodo MIN
            best_val = math.inf
            for action in actions:
                child = board.clone()
                child.play(action, current_player)
                _, val = self._minimax(child, 3 - current_player, depth - 1, alpha, beta)
                if val < best_val:
                    best_val = val
                    best_action = action
                beta = min(beta, best_val)
                if beta <= alpha:
                    break  # poda alfa
            return best_action, best_val


# ==============================================================================
# Variante sin poda — usada solo para medir el impacto de Alpha-Beta
# ==============================================================================

class MinimaxNoPruningAgent(MinimaxAgent):
    """Minimax sin poda Alpha-Beta; idéntica lógica pero sin cortes."""

    def _minimax(self, board, current_player, depth, alpha, beta):
        self._nodes_visited += 1

        done, winner = board.is_end(current_player)
        if done:
            return None, math.inf if winner == self.player else -math.inf
        if depth == 0:
            return None, self.heuristic_utility(board)

        actions = board.get_possible_actions(current_player)
        best_action = actions[0] if actions else None

        if current_player == self.player:
            best_val = -math.inf
            for action in actions:
                child = board.clone()
                child.play(action, current_player)
                _, val = self._minimax(child, 3 - current_player, depth - 1, -math.inf, math.inf)
                if val > best_val:
                    best_val = val
                    best_action = action
            return best_action, best_val
        else:
            best_val = math.inf
            for action in actions:
                child = board.clone()
                child.play(action, current_player)
                _, val = self._minimax(child, 3 - current_player, depth - 1, -math.inf, math.inf)
                if val < best_val:
                    best_val = val
                    best_action = action
            return best_action, best_val