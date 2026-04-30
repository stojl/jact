import jax
import jax.numpy as jnp
from typing import Union

class ProbabilityCallbacks:

    @staticmethod
    @jax.jit
    def none(p: jnp.ndarray, p_point: jnp.ndarray) -> None:
        """Probability empty callback

        Callback ignores probabilities altogether.

        Parameters
        ----------
        p : jnp.ndarray
            A 3D array of shape (Batch, States, Duration). This argument
            is **ignored**.
        p_point : jnp.ndarray
            A 2D array of shape (Batch, Duration). This argument is 
            **ignored**.

        Returns
        -------
        None
        """
        return None

    @staticmethod
    @jax.jit
    def default(
        p: jnp.ndarray, p_point: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Probability default callback

        Returns the absolutely continuous part and the point mass of the
        transition probabilities.

        Parameters
        ----------
        p : jnp.ndarray
            A 3D array of shape (Batch, States, Duration)
        p_point : jnp.ndarray
            A 2D array of shape (Batch, Duration)

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            The absolutely continuous part (p) and point mass (p_point).
        """
        return p, p_point

    @staticmethod
    @jax.jit
    def no_duration(
        p: jnp.ndarray, p_point: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Probability with marginalized duration callback

        Returns the absolutely continuous part and the point mass of the
        transition probabilities and marginalizes duration.

        Parameters
        ----------
        p : jnp.ndarray
            A 3D array of shape (Batch, States, Duration)
        p_point : jnp.ndarray
            A 2D array of shape (Batch, Duration)

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            The absolutely continuous part (p) of shape (Batch, States) and
            point mass (p_point) of shape (Batch,).
        """
        return p[..., -1], p_point[..., -1]

    @staticmethod
    @jax.jit
    def collapse_point(p: jnp.ndarray, p_point: jnp.ndarray) -> jnp.ndarray:
        """Probability collapse point mass callback

        Computes the sum of the absolutely continuous part and the point mass
        of the transition probabilities.

        Parameters
        ----------
        p : jnp.ndarray
            A 3D array of shape (Batch, States, Duration)
        p_point : jnp.ndarray
            A 2D array of shape (Batch, Duration)

        Returns
        -------
        jnp.ndarray
            The sum of the absolutely continuous part (p) and point
            mass (p_point).
        """
        p_with_point = p[..., 0, :] + p_point
        p = p.at[..., 0, :].set(p_with_point)
        return p

    @staticmethod
    @jax.jit
    def collapse_point_no_duration(p: jnp.ndarray, p_point: jnp.ndarray) -> jnp.ndarray:
        """Probability collapse point mass with marginalized duration callback

        Computes the sum of the absolutely continuous part and the point mass
        of the transition probabilities and marginalizes the duration.

        Parameters
        ----------
        p : jnp.ndarray
            A 3D array of shape (Batch, States, Duration)
        p_point : jnp.ndarray
            A 2D array of shape (Batch, Duration). This argument is 
            **ignored**.

        Returns
        -------
        jnp.ndarray
            The sum of the absolutely continuous part (p) and point
            mass (p_point) and marginalizes the duration as an array of shape
            (Batch, States).
        """
        p = ProbabilityCallbacks.collapse_point(p, p_point)
        return p[..., -1]

    @staticmethod
    @jax.jit
    def point_only(p: jnp.ndarray, p_point: jnp.ndarray) -> jnp.ndarray:
        """Probability point mass only callback

        Returns only the point mass of the transition probabilities.

        Parameters
        ----------
        p : jnp.ndarray
            A 3D array of shape (Batch, States, Duration). This argument is 
            **ignored**.
        p_point : jnp.ndarray
            A 2D array of shape (Batch, Duration)

        Returns
        -------
        jnp.ndarray
            The point mass probability (p_point).
        """
        return p_point

    @staticmethod
    @jax.jit
    def point_only_no_duration(p: jnp.ndarray, p_point: jnp.ndarray) -> jnp.ndarray:
        """Probability point mass only with marginalized duration callback

        Returns only the point mass of the transition probabilities and
        marginalizes the duration.

        Parameters
        ----------
        p : jnp.ndarray
            A 3D array of shape (Batch, States, Duration). This argument is 
            **ignored**.
        p_point : jnp.ndarray
            A 2D array of shape (Batch, Duration)

        Returns
        -------
        jnp.ndarray
            The point mass probability (p_point) with marginalized
            duration as an array of shape (Batch,).
        """
        return p_point[..., -1]

    @staticmethod
    @jax.jit
    def no_point(p: jnp.ndarray, p_point: jnp.ndarray) -> jnp.ndarray:
        """Probability absolutely continuous callback

        Returns only the absolutely continuous part of the transition
        probabilities.

        Parameters
        ----------
        p : jnp.ndarray
            A 3D array of shape (Batch, States, Duration)
        p_point : jnp.ndarray
            A 2D array of shape (Batch, Duration). This argument is 
            **ignored**.

        Returns
        -------
        jnp.ndarray
            The absolutely continuous part (p).
        """
        return p

    @staticmethod
    @jax.jit
    def no_point_no_duration(p: jnp.ndarray, p_point: jnp.ndarray) -> jnp.ndarray:
        """Probability absolutely continuous with marginalized duration callback

        Returns only the absolutely continuous part of the transition
        probabilities and marginalizes the duration.

        Parameters
        ----------
        p : jnp.ndarray
            A 3D array of shape (Batch, States, Duration)
        p_point : jnp.ndarray
            A 2D array of shape (Batch, Duration). This argument is 
            **ignored**.

        Returns
        -------
        jnp.ndarray
            The absolutely continuous part (p) with marginalized
            duration as an array of shape (Batch, States).
        """
        return p[..., -1]

    @staticmethod
    def from_str(str: Union[str, None]):
        match str:
            case "default":
                return ProbabilityCallbacks.default
            case "none":
                return ProbabilityCallbacks.none
            case "no_duration":
                return ProbabilityCallbacks.no_duration
            case "collapse_point":
                return ProbabilityCallbacks.collapse_point
            case "collapse_point_no_duration":
                return ProbabilityCallbacks.collapse_point_no_duration
            case "point_only":
                return ProbabilityCallbacks.point_only
            case "point_only_no_duration":
                return ProbabilityCallbacks.point_only_no_duration
            case "no_point":
                return ProbabilityCallbacks.no_point
            case "no_point_no_duration":
                return ProbabilityCallbacks.no_point_no_duration
            case None:
                return ProbabilityCallbacks.none
            case _:
                return ProbabilityCallbacks.default
